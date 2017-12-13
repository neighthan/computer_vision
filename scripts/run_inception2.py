from argparse import ArgumentParser
import os
import numpy as np
import tensorflow as tf
import tflearn
from utils import load_data, get_abs_path
from gpu_dashboard.tf_utils import tf_init
import time
from models import CNN
from layers import BranchedLayer, AvgPoolLayer, MergeLayer, LayerModule, MaxPoolLayer, GlobalAvgPoolLayer,\
    FlattenLayer, DropoutLayer, ConvLayer as _ConvLayer
from typing import Union, Sequence, Dict, Optional
_numeric = Union[int, float]
_OneOrMore = lambda type_: Union[type_, Sequence[type_]]


def acc_at_k(k: int, preds, labels: np.ndarray) -> float:
    top_k_preds = np.stack(preds.apply(lambda row: row.sort_values(ascending=False)[:k].index.tolist(), axis=1))
    acc = np.mean([labels[i] in top_k_preds[i] for i in range(len(labels))])
    return acc


def augment_imgs(imgs: np.ndarray, train: bool=False, rotate_angle: int = 10, shear_intensity: float = .15, width_shift_frac: float = .1,
                 height_shift_frac: float = .1, width_zoom_frac: float = .85, height_zoom_frac: float = .85,
                 crop_height: int = 100, crop_width: int = 100) -> np.ndarray:
    keras_params = dict(row_axis=0, col_axis=1, channel_axis=2, fill_mode='reflect')

    rotate = lambda img: tf.keras.preprocessing.image.random_rotation(img, rotate_angle, **keras_params)
    shear = lambda img: tf.keras.preprocessing.image.random_shear(img, shear_intensity, **keras_params)
    # shift = lambda img: tf.keras.preprocessing.image.random_shift(img, width_shift_frac, height_shift_frac, **keras_params)
    # zoom = lambda img: tf.keras.preprocessing.image.random_zoom(img, (width_zoom_frac, height_zoom_frac), **keras_params)

    img_aug = tflearn.data_augmentation.ImageAugmentation()
    img_aug.add_random_crop((crop_height, crop_width))
    if train:
        img_aug.add_random_flip_leftright()

    if train:
        aug_imgs = np.zeros_like(imgs)
        for i in range(len(imgs)):
            aug_imgs[i] = np.random.choice([rotate, shear])(imgs[i])
    else:
        aug_imgs = imgs

    aug_imgs = img_aug.apply(aug_imgs)
    return aug_imgs


def train_img_generator(imgs, labels, batch_size: int=128):
    imgs = imgs['default'] if type(imgs) == dict else imgs
    labels = labels['default'] if type(labels) == dict else labels

    idx = list(range(len(labels)))
    np.random.shuffle(idx)
    i = 0
    while True:
        batch_idx = idx[i * batch_size: (i + 1) * batch_size]
        yield augment_imgs(imgs[batch_idx], train=True), labels[batch_idx]


def test_img_generator(imgs, labels, batch_size: int=128):
    imgs = imgs['default'] if type(imgs) == dict else imgs
    labels = labels['default'] if type(labels) == dict else labels

    idx = list(range(len(labels)))
    np.random.shuffle(idx)
    i = 0
    while True:
        batch_idx = idx[i * batch_size: (i + 1) * batch_size]
        yield augment_imgs(imgs[batch_idx], train=False), labels[batch_idx]


def _batch(self, tensors: _OneOrMore(tf.Tensor), inputs: Dict[str, np.ndarray],
           labels: Optional[Dict[str, np.ndarray]] = None,
           range_=None, idx: Sequence[int] = None, return_all_data: bool = True, is_training: bool = False,
           dataset: bool = False, generator=None):
    """

    :param tensors:
    :param inputs:
    :param labels:
    :param range_:
    :param idx:
    :param return_all_data: if true, the return values from each batch of the input are put into a list which is
                            returned; each element will be a list of the returned values for one given tensor. If only
                            one tensor was run, the list is still nested. If return_all_data is False, only the
                            values from running the tensors on the last batch of data will be returned; this will be
                            a list or tuple. Returning only the final value is useful for streaming metrics
    :param is_training: whether the model is currently being trained; used by, e.g., dropout and batchnorm
    :param dataset: whether the model uses the tensorflow Dataset class. If so, self.data_init_op will be run with
                    inputs, labels fed in. Otherwise, batches of inputs, labels will be fed in separately each time
                    the tensors are run. Either way, is_training will be fed in at each batch.
    :returns:
    """

    if type(tensors) not in (list, tuple):
        tensors = [tensors]

    if dataset:
        self.sess.run(self.data_init_op, self._get_feed_dict(inputs, labels))

    if idx is None:
        idx = list(range(len(next(iter(inputs.values())))))
    if range_ is None:
        range_ = range(int(np.ceil(len(idx) / self.batch_size)))

    try:
        self.sess.run(self.local_init)
    except AttributeError:  # no local_init unless using streaming metrics
        pass

    if return_all_data:
        ret = [[] for _ in range(len(tensors))]

    if generator is not None:
        generator = generator(inputs, labels)

    for batch in range_:
        feed_dict = {self.is_training: is_training}

        if not dataset:
            if generator is not None:
                inputs, labels = next(generator)
                feed_dict.update(
                    {self.inputs_p['default']: inputs, self.labels_p['default']: labels})
            else:
                feed_dict.update({self.inputs_p[name]: inputs[name][batch_idx] for name in inputs})
                if labels is not None:
                    feed_dict.update({self.labels_p[name]: labels[name][batch_idx] for name in labels})

        vals = self.sess.run(tensors, feed_dict)
        if return_all_data:
            for i in range(len(tensors)):
                ret[i].append(vals[i])

    if return_all_data:
        return ret
    else:
        return vals


def train(self, train_inputs: Union[np.ndarray, Dict[str, np.ndarray]],
          train_labels: Union[np.ndarray, Dict[str, np.ndarray]],
          dev_inputs: Union[np.ndarray, Dict[str, np.ndarray]], dev_labels: Union[np.ndarray, Dict[str, np.ndarray]],
          n_epochs: int = 100, max_patience: int = 5, verbose: int = 0, train_generator=None, test_generator=None):
    """
    The best epoch is the one where the early stop metric on the dev set is the highest. "best" in reference to other
    metrics means the value of that metric at the best epoch.
    :param train_inputs:
    :param train_labels:
    :param dev_inputs:
    :param dev_labels:
    :param n_epochs:
    :param max_patience:
    :param verbose: 3 for tnrange, 2 for trange, 1 for range w/ print, 0 for range
    :returns: {name: value} of the various metrics at the best epoch; includes train_time and whether training was
              completed
    """

    start_time = time.time()

    if verbose == 3:
        epoch_range = lambda *args: tnrange(*args, unit='epoch')
        batch_range = lambda *args: tnrange(*args, unit='batch', leave=False)
    elif verbose == 2:
        epoch_range = lambda *args: trange(*args, unit='epoch')
        batch_range = lambda *args: trange(*args, unit='batch', leave=False)
    else:
        epoch_range = range
        batch_range = range

    train_inputs, train_labels, dev_inputs, dev_labels = [x if type(x) is dict else {'default': x}
                                                          for x in [train_inputs, train_labels, dev_inputs, dev_labels]]

    if self._metric_improved(0, 1):  # higher is better; start low
        best_early_stop_metric = -np.inf
    else:
        best_early_stop_metric = np.inf

    patience = max_patience
    train_idx = list(range(len(next(iter(train_labels.values())))))
    dev_idx = list(range(len(next(iter(dev_labels.values())))))
    train_batches_per_epoch = int(np.ceil(len(train_idx) / self.batch_size))
    dev_batches_per_epoch = int(np.ceil(len(dev_idx) / self.batch_size))

    metric_names = list(self.metrics.keys())
    metric_ops = [self.metrics[name] for name in metric_names] + [self.loss_op]

    epochs = epoch_range(n_epochs)
    for epoch in epochs:
        np.random.shuffle(train_idx)

        batches = batch_range(train_batches_per_epoch)

        ret = _batch(self, [self.loss_op, self.train_op], train_inputs, train_labels, batches, train_idx,
                     is_training=True, dataset=self.uses_dataset, generator=train_generator)
        train_loss = np.array(ret)[0, :].mean()

        batches = batch_range(dev_batches_per_epoch)
        ret = _batch(self, metric_ops, dev_inputs, dev_labels, batches, dev_idx, dataset=self.uses_dataset,
                     generator=test_generator)
        ret = np.array(ret)

        dev_loss = ret[-1, :].mean()
        dev_metrics = ret[:-1, -1]  # last values, because metrics are streaming
        dev_metrics = {metric_names[i]: dev_metrics[i] for i in range(len(metric_names))}
        dev_metrics.update({'dev_loss': dev_loss})
        early_stop_metric = dev_metrics[self.early_stop_metric_name]

        if verbose == 1:
            print(f"Train loss: {train_loss:.3f}; Dev loss: {dev_loss:.3f}. Metrics: {dev_metrics}")

        if self.record:
            self._add_summaries(epoch, {'loss': train_loss}, dev_metrics)

        if self._metric_improved(best_early_stop_metric, early_stop_metric):  # always keep updating the best model
            train_time = (time.time() - start_time) / 60  # in minutes
            best_metrics = dev_metrics
            best_metrics.update({'train_loss': train_loss, 'train_time': train_time, 'train_complete': False})
            if self.record:
                try:
                    self._log(best_metrics)
                except:
                    pass
                self._save()

        if self._metric_improved(best_early_stop_metric, early_stop_metric, significant=True):
            best_early_stop_metric = early_stop_metric
            patience = max_patience
        else:
            patience -= 1
            if patience == 0:
                break

        runtime = (time.time() - start_time) / 60
        if verbose > 1:
            epochs.set_description(
                f"Epoch {epoch + 1}. Train Loss: {train_loss:.3f}. Dev loss: {dev_loss:.3f}. Runtime {runtime:.2f}.")

    best_metrics['train_complete'] = True
    if self.record:
        self._log(best_metrics)
        self.saver.restore(self.sess, os.path.join(self.log_dir, 'model.ckpt'))  # reload best epoch

    return best_metrics


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, required=True)
    parser.add_argument('-l2', '--l2_lambda', type=float)
    parser.add_argument('-s', '--scaling', action='store_true')
    parser.add_argument('-na', type=int, required=True)
    parser.add_argument('-nb', type=int, required=True)
    parser.add_argument('-nc', type=int, required=True)
    parser.add_argument('-bn', '--batch_norm', action='store_true')
    parser.add_argument('-p', '--drop_prob', type=float)
    parser.add_argument('-d', '--device', type=int, required=True)

    args = parser.parse_args()
    batch_norm = args.batch_norm
    config = tf_init(args.device)

    def ConvLayer(*args, **kwargs):
        return _ConvLayer(*args, batch_norm=batch_norm, **kwargs)

    train_inputs, train_labels, val_inputs, val_labels, test_inputs = load_data('miniplaces')
    n_classes = len(np.unique(train_labels))

    inception_a = LayerModule([
        BranchedLayer([AvgPoolLayer(1, 1), ConvLayer(96, 1), ConvLayer(64, 1), ConvLayer(64, 1)]),
        BranchedLayer([ConvLayer(96, 1), None, ConvLayer(96, 3), ConvLayer(96, 3)]),
        BranchedLayer([None, None, None, ConvLayer(96, 3)]),
        MergeLayer(axis=3)
    ])

    inception_b = LayerModule([
        BranchedLayer([AvgPoolLayer(1, 1), ConvLayer(384, 1), ConvLayer(192, 1), ConvLayer(192, 1)]),
        BranchedLayer([ConvLayer(128, 1), None, ConvLayer(224, [7, 1]), ConvLayer(192, [1, 7])]),
        BranchedLayer([None, None, ConvLayer(256, [1, 7]), ConvLayer(224, [7, 1])]),
        BranchedLayer([None, None, None, ConvLayer(224, [1, 7])]),
        BranchedLayer([None, None, None, ConvLayer(256, [7, 1])]),
        MergeLayer(axis=3)
    ])

    inception_c = LayerModule([
        BranchedLayer([AvgPoolLayer(1, 1), ConvLayer(256, 1), ConvLayer(384, 1), ConvLayer(384, 1)]),
        BranchedLayer([ConvLayer(256, 1), None, BranchedLayer([ConvLayer(256, [1, 3]), ConvLayer(256, [3, 1])]), ConvLayer(448, [1, 3])]),
        BranchedLayer([None, None, None, ConvLayer(512, [3, 1])]),
        BranchedLayer([None, None, None, BranchedLayer([ConvLayer(256, [3, 1]), ConvLayer(256, [1, 3])])]),
        MergeLayer(axis=3)
    ])

    layers = [
        ConvLayer(32, 3, 2, padding='valid'),
        ConvLayer(32, 3),
        ConvLayer(64, 3),
        # don't use stride of 2 since our images are smaller
        BranchedLayer([MaxPoolLayer(3, padding='valid'), ConvLayer(96, 3, padding='valid')]),
        MergeLayer(axis=3),
        BranchedLayer([ConvLayer(64, 1), ConvLayer(64, 1)]),
        BranchedLayer([ConvLayer(96, 3, padding='valid'), ConvLayer(64, [7, 1])]),
        BranchedLayer([None, ConvLayer(64, [1, 7])]),
        BranchedLayer([None, ConvLayer(96, 3, padding='valid')]),
        MergeLayer(axis=3),
        BranchedLayer([ConvLayer(192, 3, strides=2, padding='valid'), MaxPoolLayer(3, strides=2, padding='valid')]),
        MergeLayer(axis=3),
        *([inception_a] * args.na),  # x4
        ConvLayer(1024, 3, strides=2),  # reduction_a
        *([inception_b] * args.nb),  # x7
        ConvLayer(1536, 3, strides=2),  # reduction_b
        *([inception_c] * args.nc),  # x3
        GlobalAvgPoolLayer(),
        FlattenLayer(),
        DropoutLayer(rate=args.drop_prob)
    ]

    data_params = {'na': args.na, 'nb': args.nb, 'nc': args.nc, 'batch_norm': batch_norm, 'drop_prob': args.drop_prob,
                   'augmentation': True}

    cnn = CNN(layers, n_classes=n_classes, batch_size=128, l2_lambda=args.l2_lambda,
              learning_rate=args.learning_rate, add_scaling=args.scaling, data_params=data_params,
              models_dir=get_abs_path('../miniplaces/models'), config=config,
              img_width=100, img_height=100)
    train(cnn, train_inputs, train_labels, val_inputs, val_labels, verbose=1,
          train_generator=train_img_generator, test_generator=test_img_generator)

    # can't do this part; images are the wrong size (need to be 100 x 100 now)
    # preds = cnn.predict_proba(val_inputs)
    # acc1 = acc_at_k(1, preds, val_labels)
    # acc5 = acc_at_k(5, preds, val_labels)
    # print(f'Args: {args}; acc1: {acc1}; acc5: {acc5}')

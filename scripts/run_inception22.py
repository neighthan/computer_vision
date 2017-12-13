from argparse import ArgumentParser
import os
import numpy as np
import tensorflow as tf
import tflearn
from computer_vision.scripts.utils import load_data, get_abs_path, tf_init
import time
from computer_vision.scripts.models import CNN2
from computer_vision.scripts.layers import BranchedLayer, AvgPoolLayer, MergeLayer, LayerModule, MaxPoolLayer, GlobalAvgPoolLayer,\
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
    aug = 'LR_contrast_.75_hue_.15_rot_4'
    model_name = f"inception_aug_{aug}_l2_{args.l2_lambda}_p_{args.drop_prob}_na_{args.na}_nb_{args.nb}_nc_{args.nc}" \
                 f"_bn_before_{args.batch_norm}_lr_{args.learning_rate}_e700_bs64"

    def ConvLayer(*args, **kwargs):
        return _ConvLayer(*args, batch_norm=batch_norm, **kwargs)

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

    train_fnames = ['/afs/csail.mit.edu/u/n/nhunt/github/computer_vision/miniplaces/data/train_0.tfrecord']
    dev_fnames = ['/afs/csail.mit.edu/u/n/nhunt/github/computer_vision/miniplaces/data/val_0.tfrecord']

    cnn = CNN2(layers, n_classes=100, batch_size=64, l2_lambda=args.l2_lambda,
              learning_rate=args.learning_rate, add_scaling=args.scaling, data_params=data_params,
              models_dir=get_abs_path('../miniplaces/models'), config=config,
              img_width=128, img_height=128, augment=2, model_name=model_name)
    cnn.train(train_fnames, dev_fnames, train_batches_per_epoch=700, dev_batches_per_epoch=int(10_000 / 128), verbose=1,
              max_patience=10)

    # can't do this part; images are the wrong size (need to be 100 x 100 now)
    # preds = cnn.predict_proba(val_inputs)
    # acc1 = acc_at_k(1, preds, val_labels)
    # acc5 = acc_at_k(5, preds, val_labels)
    # print(f'Args: {args}; acc1: {acc1}; acc5: {acc5}')

import numpy as np
import pandas as pd
import tensorflow as tf
from tflearn import activations
from tqdm import tnrange, trange
import time
import os
import sys
from utils import tf_init, get_next_run_num, get_abs_path
from typing import List, Optional, Dict, Any, Union, Sequence
from layers import ConvLayer, MaxPoolLayer, AvgPoolLayer, BranchedLayer, MergeLayer, LayerModule, FlattenLayer,\
    DenseLayer, DropoutLayer, GlobalAvgPoolLayer, GlobalMaxPoolLayer, LSTMLayer, _Layer
import warnings

_numeric = Union[int, float]
_OneOrMore = lambda type_: Union[type_, Sequence[type_]]

tf.logging.set_verbosity(tf.logging.WARN)
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

_cnn_modules = {
    'vgg16': tf.contrib.keras.applications.VGG16,
    'xception': tf.contrib.keras.applications.Xception
}

_activations = {
    'relu': tf.nn.relu,
    'prelu': activations.prelu
}


class BaseNN(object):
    """
    This class implements several methods that may be used by neural networks in general. It doesn't actually create any
    layers, so it shouldn't be used directly.
    Currently, this class only works for classification, not regression. It also defines the score method already and expects that
    the model can calculate accuracy@1 and accuracy@5, so it expects >= 5-way classification.
    """

    def __init__(
        self,
        layers: Optional[List[_Layer]]        = None,
        models_dir:                       str = '',
        log_key:                          str = 'default',
        n_regress_tasks:                  int = 0,
        n_classes:            _OneOrMore(int) = (),
        task_names:   Optional[Sequence[str]] = None,
        config:      Optional[tf.ConfigProto] = None,
        run_num:                          int = -1,
        batch_size:                       int = 128,
        record:                          bool = True,
        random_state:                     int = 521,
        data_params: Optional[Dict[str, Any]] = None
    ):

        # don't include run_num or config
        param_names = ['layers', 'n_classes', 'n_regress_tasks', 'task_names', 'random_state', 'batch_size', 'data_params']

        if record:
            assert models_dir, "models_dir must be specifed to record a model."

        self.log_fname = f'{models_dir}/log.h5'

        if config is None:
            config = tf_init()

        if data_params is None:
            data_params = {}

        if type(n_classes) == int:
            n_classes = (n_classes,)

        if len(n_classes) == 0 and n_regress_tasks == 0:
            n_regress_tasks = 1 # assume doing (a single) regression if n_classes isn't given

        if task_names is None:
            if n_regress_tasks + len(n_classes) > 1:
                raise AttributeError('task_names must be specified for a multi-task model.')
            else: # to make single task easier to use; just set a name
                task_names = ('default',)

        if run_num == -1:
            assert type(layers) is not None
            assert n_regress_tasks > 0 or len(n_classes) > 0
            self.run_num         = get_next_run_num(f'{models_dir}/run_num.pkl') if record else 0
            self.layers          = layers
            self.n_regress_tasks = n_regress_tasks
            self.n_classes       = n_classes
            self.task_names      = task_names
            self.random_state    = random_state
            self.batch_size      = batch_size
            self.record          = record
            self.data_params     = data_params
        else:
            assert models_dir, "models_dir must be specified to reload a model."
            self.run_num = run_num
            log = pd.read_hdf(self.log_fname, log_key)
            for param in param_names:
                p = log.loc[run_num, param]
                self.__setattr__(param, p if type(p) != np.float64 else p.astype(np.float32))

        self.config        = config
        self.models_dir    = models_dir
        self.log_key       = log_key
        self.n_class_tasks = len(self.n_classes)
        self.params        = {param: self.__getattribute__(param) for param in param_names}

        if record:
            self.log_dir = f'{models_dir}/{self.run_num}/'
            try:
                os.makedirs(self.log_dir)
            except OSError: # already exists if loading saved model
                pass
        else:
            self.log_dir = ''

        tf.set_random_seed(self.random_state)
        np.random.seed(self.random_state)

    def _check_graph(self):
        required_attrs = ['loss_op', 'train_op', 'inputs_p', 'labels_p', 'predict', 'metrics', 'early_stop_metric_name',
                          'is_training', 'uses_dataset']
        for attr in required_attrs:
            try:
                getattr(self, attr)
            except AttributeError:
                print(f"Missing required attribute {attr}.")
                raise

    def _build_graph(self):
        """
        This method should be overridden by all subclasses. The code below is just starter code.
        """

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            self.global_init = tf.global_variables_initializer()
            self.is_training = tf.placeholder_with_default(False, [])

            self.metrics = {} # {'name': tensor}; must be able to calculate on a stream, at least for now
            self.early_stop_metric_name = 'dev_loss' # or 'auc' or...; must define a metric that matches this
            self.uses_dataset = False

        self._add_savers_and_writers()
        self._check_graph()
    
    def _add_savers_and_writers(self):
        if self.log_dir != '':
            with self.graph.as_default():
                self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'train'), self.graph, flush_secs=30)
                self.dev_writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'dev'), self.graph, flush_secs=30)
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.name, var)
                self.summary_op = tf.summary.merge_all()

        self.sess = tf.Session(graph=self.graph, config=self.config)
        
        meta_graph_file = os.path.join(self.log_dir, 'model.ckpt.meta')
        if os.path.isfile(meta_graph_file):  # load saved model
            print(f"Loading graph from: {meta_graph_file}.")
            saver = tf.train.import_meta_graph(meta_graph_file)
            saver.restore(self.sess, os.path.join(self.log_dir, 'model.ckpt'))
        else:
            self.sess.run(self.global_init)

    @staticmethod
    def _metric_improved(old_metric: _numeric, new_metric: _numeric, significant: bool=False, threshold: float=.01) -> bool:
        """
        By default, improving is *increasing*.
        Override this if you need a decrease in the metric to be an improvement (to use '<' and '-' instead of '>' and '+'
        :param old_metric:
        :param new_metric:
        :param significant:
        :param threshold:
        :return:
        """

        if significant:
            return new_metric > (1 + threshold) * old_metric
        else:
            return new_metric > old_metric

    def _get_feed_dict(self, inputs: Dict[str, np.ndarray], labels: Optional[Dict[str, np.ndarray]]=None) ->\
            Dict[tf.placeholder, np.ndarray]:
        """

        :param inputs:
        :param labels:
        :returns:
        """

        feed_dict = {self.inputs_p[name]: inputs[name] for name in inputs}
        if labels is not None:
            feed_dict.update({self.labels_p[name]: labels[name] for name in labels})
        return feed_dict

    def _batch(self, tensors: _OneOrMore(tf.Tensor), inputs: Dict[str, np.ndarray], labels: Optional[Dict[str, np.ndarray]]=None,
               range_=None, idx: Sequence[int]=None, return_all_data: bool=True, is_training: bool=False, dataset: bool=False):
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

        if not dataset and idx is None:
            idx = list(range(len(next(iter(inputs.values())))))
        if range_ is None:
            range_ = range(int(np.ceil(len(idx) / self.batch_size)))

        try:
            self.sess.run(self.local_init)
        except AttributeError:  # no local_init unless using streaming metrics
            pass

        if return_all_data:
            ret = [[] for _ in range(len(tensors))]

        for batch in range_:
            feed_dict = {self.is_training: is_training}

            if not dataset:
                batch_idx = idx[batch * self.batch_size: (batch + 1) * self.batch_size]
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

    def _add_summaries(self, epoch: int, train_vals: Dict[str, Union[int, float]], dev_vals: Dict[str, Union[int, float]]):
        summary_str = self.sess.run(self.summary_op)
        self.train_writer.add_summary(summary_str, epoch)

        for val_name in train_vals:
            self.train_writer.add_summary(tf.Summary(
                value=[tf.Summary.Value(tag=val_name, simple_value=train_vals[val_name])]), epoch)

        for val_name in dev_vals:
            self.dev_writer.add_summary(tf.Summary(
                value=[tf.Summary.Value(tag=val_name, simple_value=dev_vals[val_name])]), epoch)

    def _log(self, extras):
        params_df = (pd.DataFrame([[self.params[key] for key in self.params.keys()]], columns=self.params.keys(),
                                  index=[self.run_num])
                     .assign(**extras))

        try:
            log = pd.read_hdf(self.log_fname, self.log_key)

            if self.run_num in log.index:
                log = log.drop(self.run_num)
            pd.concat([log, params_df]).to_hdf(self.log_fname, self.log_key)
        except (IOError, KeyError):  # this file or key doesn't exist yet
            params_df.to_hdf(self.log_fname, self.log_key)

    def _save(self):
        self.saver.save(self.sess, os.path.join(self.log_dir, 'model.ckpt'))

    def train(self, train_inputs: Union[np.ndarray, Dict[str, np.ndarray]], train_labels: Union[np.ndarray, Dict[str, np.ndarray]],
              dev_inputs: Union[np.ndarray, Dict[str, np.ndarray]], dev_labels: Union[np.ndarray, Dict[str, np.ndarray]],
              n_epochs: int=100, max_patience: int=5, verbose: int=0):
        """
        The best epoch is the one where the early stop metric on the dev set is the highest. "best" in reference to other
        metrics means the value of that metric at the best epoch.
        :param train_inputs:
        :param train_labels:
        :param dev_inputs:
        :param dev_labels:
        :param n_epochs:
        :param max_patience:
        :param verbose: 2 for tnrange, 1 for trange, 0 for range
        :returns: {name: value} of the various metrics at the best epoch; includes train_time and whether training was
                  completed
        """

        start_time = time.time()

        if verbose == 2:
            epoch_range = lambda *args: tnrange(*args, unit='epoch')
            batch_range = lambda *args: tnrange(*args, unit='batch', leave=False)
        elif verbose == 1:
            epoch_range = lambda *args: trange(*args, unit='epoch')
            batch_range = lambda *args: trange(*args, unit='batch', leave=False)
        else:
            epoch_range = range
            batch_range = range

        train_inputs, train_labels, dev_inputs, dev_labels = [x if type(x) is dict else {'default': x}
                                                              for x in [train_inputs, train_labels, dev_inputs, dev_labels]]

        if self._metric_improved(0, 1): # higher is better; start low
            best_early_stop_metric = -np.inf
        else:
            best_early_stop_metric = np.inf

        patience = max_patience
        train_idx = list(range(len(next(iter(train_labels.values())))))
        dev_idx = list(range(len(next(iter(dev_labels.values())))))
        train_batches_per_epoch = int(np.ceil(len(train_idx) / self.batch_size))
        dev_batches_per_epoch   = int(np.ceil(len(dev_idx) / self.batch_size))

        metric_names = list(self.metrics.keys())
        metric_ops = [self.metrics[name] for name in metric_names]+ [self.loss_op]

        epochs = epoch_range(n_epochs)
        for epoch in epochs:
            np.random.shuffle(train_idx)

            batches = batch_range(train_batches_per_epoch)

            ret = self._batch([self.loss_op, self.train_op], train_inputs, train_labels, batches, train_idx,
                                 is_training=True, dataset=self.uses_dataset)
            train_loss = np.array(ret)[:, 0].mean()

            batches = batch_range(dev_batches_per_epoch)
            ret = self._batch(metric_ops, dev_inputs, dev_labels, batches, dev_idx, dataset=self.uses_dataset)

            dev_loss = np.array(ret)[:, -1].mean()
            dev_metrics = ret[-1][:-1] # last values, because metrics are streaming
            dev_metrics = {metric_names[i]: dev_metrics[i] for i in range(len(metric_names))}
            dev_metrics.update({'dev_loss': dev_loss})
            early_stop_metric = dev_metrics[self.early_stop_metric_name]

            if self.record:
                self._add_summaries(epoch, {'loss': train_loss}, dev_metrics)

            if self._metric_improved(best_early_stop_metric, early_stop_metric): # always keep updating the best model
                train_time = (time.time() - start_time) / 60 # in minutes
                best_metrics = dev_metrics
                best_metrics.update({'train_loss': train_loss, 'train_time': train_time, 'train_complete': False})
                if self.record:
                    self._log(best_metrics)
                    self._save()

            if self._metric_improved(best_early_stop_metric, early_stop_metric, significant=True):
                best_early_stop_metric = early_stop_metric
                patience = max_patience
            else:
                patience -= 1
                if patience == 0:
                    break

            runtime = (time.time() - start_time) / 60
            if verbose:
                epochs.set_description(
                    f"Epoch {epoch + 1}. Train Loss: {train_loss:.3f}. Dev loss: {dev_loss:.3f}. Runtime {runtime:.2f}.")

        if self.record:
            best_metrics['train_complete'] = True
            self._log(best_metrics)
            self.saver.restore(self.sess, os.path.join(self.log_dir, 'model.ckpt')) # reload best epoch

        return best_metrics

    def predict_proba(self, inputs: Union[np.ndarray, Dict[str, np.ndarray]]) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Generates predictions (predicted 'probabilities', not binary labels)
        :returns: array of predicted probabilities of being positive for each sample in the test set
        """

        if type(inputs) is not dict:
            inputs = {'default': inputs}

        predictions = self._batch([self.predict[name] for name in self.task_names], inputs, dataset=self.uses_dataset) # pd.DataFrame
        predictions = {name: pd.DataFrame(np.concatenate(predictions[i])) for name, i in enumerate(self.task_names)}

        if len(predictions.keys()) == 1 and next(iter(predictions)) == 'default':
            return predictions['default']

        return predictions

    def score(self, inputs, labels):
        raise NotImplemented


# class PretrainedCNN(BaseNN):
#     """
#     """
#     def __init__(
#         self,
#         img_width        = 128,
#         img_height       = 128,
#         n_channels       = 3,
#         n_classes        = None,
#         log_fname        = f'{miniplaces}/models/log.h5',
#         log_key          = 'default',
#         data_params      = None,
#         dense_nodes      = (128,),
#         l2_lambda        = None,
#         learning_rate    = 0.001,
#         beta1            = 0.9,
#         beta2            = 0.999,
#         config           = None,
#         run_num          = -1,
#         batch_size       = 64,
#         record           = True,
#         random_state     = 521,
#         dense_activation = 'relu',
#         finetune         = False,
#         pretrained_weights = False,
#         cnn_module       = 'vgg16'
#     ):
#         new_run = run_num == -1
#         super(PretrainedCNN, self).__init__(log_fname, log_key, config, run_num, batch_size, record, random_state)
#
#         param_names = ['img_width', 'img_height', 'n_channels', 'n_classes', 'dense_nodes', 'l2_lambda', 'learning_rate',
#                        'beta1', 'beta2', 'dense_activation', 'finetune', 'cnn_module', 'pretrained_weights']
#
#         if not pretrained_weights:
#             finetune = True
#
#         if data_params is None:
#             data_params = {}
#
#         if new_run:
#             assert type(n_classes) is not None
#             self.img_height = img_height
#             self.img_width = img_width
#             self.n_channels = n_channels
#             self.n_classes = n_classes
#             self.dense_nodes = dense_nodes
#             self.l2_lambda = l2_lambda
#             self.learning_rate = learning_rate
#             self.beta1 = beta1
#             self.beta2 = beta2
#             self.dense_activation = dense_activation
#             self.finetune = finetune
#             self.cnn_module = cnn_module
#             self.pretrained_weights = pretrained_weights
#         else:
#             log = pd.read_hdf(log_fname, log_key)
#             for param in param_names:
#                 p = log.loc[run_num, param]
#                 self.__setattr__(param, p if type(p) != np.float64 else p.astype(np.float32))
#
#         self.params.update(data_params)
#         self.params.update({param: self.__getattribute__(param) for param in param_names})
#
#         self._build_graph()
#
#     def _build_graph(self):
#         """
#         If self.log_dir contains a previously trained model, then the graph from that run is loaded for further
#         training/inference. Otherwise, a new graph is built.
#         Also starts a session with self.graph.
#         If self.log_dir != '' then a Saver and summary writers are also created.
#         :returns: None
#         """
#
#         self.graph = tf.Graph()
#         self.sess = tf.Session(config=self.config, graph=self.graph)
#
#         with self.graph.as_default(), self.sess.as_default():
#             self.inputs_p = tf.placeholder(tf.float32, shape=(None, self.img_height, self.img_width, self.n_channels), name='inputs_p')
#             self.labels_p = tf.placeholder(tf.int32, shape=None, name='labels_p')
#             self.is_training = tf.placeholder_with_default(False, [])
#
#             with tf.variable_scope('cnn'):
#                 weights = 'imagenet' if self.pretrained_weights else None
#
#                 cnn = cnn_modules[self.cnn_module]
#                 cnn_out = cnn(include_top=False, input_tensor=self.inputs_p, weights=weights).output
#                 hidden = tf.contrib.layers.flatten(cnn_out)
#
#             dense_activation = activations[self.dense_activation]
#             with tf.variable_scope('dense'):
#                 for i in range(len(self.dense_nodes)):
#                     hidden = tf.layers.dense(hidden, self.dense_nodes[i], activation=dense_activation, name='dense_{}'.format(i))
#                 self.logits = tf.layers.dense(hidden, self.n_classes, activation=None, name='logits')
#
#             self.predict = tf.nn.softmax(self.logits, name='predict')
#             self.loss_op = tf.losses.sparse_softmax_cross_entropy(self.labels_p, self.logits, scope='xent')
#
#             if self.l2_lambda:
#                 self.loss_op = tf.add(self.loss_op, self.l2_lambda * tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()]), name='loss')
#
#             trainable_vars = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#
#             if not self.finetune: # don't finetune CNN layers
#                 trainable_vars = filter(lambda tensor: not tensor.name.startswith('cnn'), trainable_vars)
#
#             self.optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2)
#             self.train_op = self.optimizer.minimize(self.loss_op, var_list=trainable_vars, name='train_op')
#
#             self.probs_p = tf.placeholder(tf.float32, shape=(None, self.n_classes), name='probs_p')
#             self.accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.probs_p, self.labels_p, k=1), tf.float32))
#             self.accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.probs_p, self.labels_p, k=5), tf.float32))
#
#             self.saver = tf.train.Saver()
#             self.global_init = tf.global_variables_initializer()
#         self._add_savers_and_writers()
#         self._check_graph()


class CNN(BaseNN):
    """

    """
    def __init__(
        self,
        layers: Optional[List[_Layer]]        = None,
        models_dir:                       str = '',
        log_key:                          str = 'default',
        n_regress_tasks:                  int = 0,
        n_classes:            _OneOrMore(int) = (),
        task_names:   Optional[Sequence[str]] = None,
        config:      Optional[tf.ConfigProto] = None,
        run_num:                          int = -1,
        batch_size:                       int = 64,
        record:                          bool = True,
        random_state:                     int = 521,
        data_params: Optional[Dict[str, Any]] = None,
        # begin class specific parameters
        img_width:                        int = 128,
        img_height:                       int = 128,
        n_channels:                       int = 3,
        l2_lambda:            Optional[float] = None,
        learning_rate:                  float = 0.001,
        beta1:                          float = 0.9,
        beta2:                          float = 0.999,
        add_scaling:                     bool = False
    ):
        new_run = run_num == -1
        super().__init__(layers, models_dir, log_key, n_regress_tasks, n_classes, task_names, config, run_num,
                         batch_size, record, random_state, data_params)

        param_names = ['img_width', 'img_height', 'n_channels', 'l2_lambda', 'learning_rate', 'beta1', 'beta2', 'add_scaling']

        if new_run:
            self.img_height    = img_height
            self.img_width     = img_width
            self.n_channels    = n_channels
            self.l2_lambda     = l2_lambda
            self.learning_rate = learning_rate
            self.beta1         = beta1
            self.beta2         = beta2
            self.add_scaling   = add_scaling
        else:
            log = pd.read_hdf(self.log_fname, log_key)
            for param in param_names:
                p = log.loc[run_num, param]
                self.__setattr__(param, p if type(p) != np.float64 else p.astype(np.float32))

        self.params.update({param: self.__getattribute__(param) for param in param_names})

        self._build_graph()


    def _build_graph(self):
        """
        If self.log_dir contains a previously trained model, then the graph from that run is loaded for further
        training/inference. Otherwise, a new graph is built.
        Also starts a session with self.graph.
        If self.log_dir != '' then a Saver and summary writers are also created.
        :returns: None
        """

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs_p = {'default': tf.placeholder(tf.float32, shape=(None, self.img_height, self.img_width, self.n_channels), name='inputs_p')}
            self.is_training = tf.placeholder_with_default(False, [])

            if self.add_scaling:
                mean, std = tf.nn.moments(self.inputs_p['default'], axes=[1, 2], keep_dims=True)
                hidden = (self.inputs_p['default'] - mean) / tf.sqrt(std)
            else:
                hidden = self.inputs_p['default']

            for layer in self.layers:
                hidden = layer.apply(hidden, is_training=self.is_training)

            self.predict = {}
            self.labels_p = {}
            self.loss_ops = {}

            if self.n_class_tasks > 0:
                self.logits = {}
                self.probs_p = {}
                self.accuracy = {}
                for i in range(self.n_class_tasks):
                    name = self.task_names[i]
                    with tf.variable_scope(f"class_{name}"):
                        self.labels_p[name] = tf.placeholder(tf.int32, shape=None, name='labels_p')
                        self.logits[name] = tf.layers.dense(hidden, self.n_classes[i], activation=None, name='logits')

                        self.predict[name] = tf.nn.softmax(self.logits[name], name='predict')
                        self.loss_ops[name] = tf.losses.sparse_softmax_cross_entropy(self.labels_p[name], self.logits[name], scope='xent')

                        self.accuracy[name] = tf.metrics.accuracy(self.labels_p[name], tf.arg_max(self.predict[name], 1))

                        # self.probs_p[name] = tf.placeholder(tf.float32, shape=(None, self.n_classes[i]), name='probs_p')
                        # self.accuracy[name] = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.probs_p[name], self.labels_p[name], k=1), tf.float32))
                        # TODO: make it so we can add a desired tensor afterwards? acc@5 shouldn't always be here.
                        # maybe just do with model.graph.as_default(): model.acc5 = ... ??
                        # self.accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.probs_p, self.labels_p, k=5), tf.float32))

            for i in range(self.n_class_tasks, self.n_class_tasks + self.n_regress_tasks):
                name = self.task_names[i]
                with tf.variable_scope(f"regress_{name}"):
                    self.labels_p[name] = tf.placeholder(tf.float32, shape=None, name='labels_p')
                    self.predict[name] = tf.layers.dense(hidden, 1, activation=None)
                    self.loss_ops[name] = tf.losses.mean_squared_error(self.labels_p[name], self.predict[name], scope='mse')

            self.loss_op = tf.add_n(list(self.loss_ops.values()))
            if self.l2_lambda:
                self.loss_op = tf.add(self.loss_op, self.l2_lambda * tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()]), name='loss')

            # self.train_op = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2).minimize(self.loss_op, name='train_op')
            self.global_step = tf.Variable(0, trainable=False)
            self.decayed_lr = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                         decay_steps=200000 // self.batch_size, decay_rate=0.94)
            self.optimizer = tf.train.RMSPropOptimizer(self.decayed_lr)#, epsilon=1)
            self.train_op = self.optimizer.minimize(self.loss_op, global_step=self.global_step, name='train_op')

            self.metrics = {f'acc_{name}': self.accuracy[name] for name in self.accuracy}
            self.early_stop_metric_name = 'dev_loss'
            self.uses_dataset = False

            self.saver = tf.train.Saver()
            self.global_init = tf.global_variables_initializer()

        self._add_savers_and_writers()
        self._check_graph()

    def score(self, inputs, labels):
        probs = self.predict_proba(inputs)
        return self.sess.run([self.accuracy1, self.accuracy5], {self.probs_p: probs, self.labels_p: labels})


class RNN(BaseNN):
    """
    """
    def __init__(
        self,
        layers:         Optional[List[_Layer]] = None,
        models_dir:                        str = '',
        log_key:                           str = 'default',
        n_regress_tasks:                   int = 0,
        n_classes:             _OneOrMore(int) = (),
        task_names:    Optional[Sequence[str]] = None,
        config:       Optional[tf.ConfigProto] = None,
        run_num:                           int = -1,
        batch_size:                        int = 256,
        record:                           bool = True,
        random_state:                      int = 1,
        data_params:  Optional[Dict[str, Any]] = None,
        # begin class specific parameters
        n_timesteps:                       int = -1,
        n_features: Union[int, Dict[str, int]] = None,
        l2_lambda:             Optional[float] = None,
        learning_rate:                   float = 0.001,
        beta1:                           float = 0.9,
        beta2:                           float = 0.999,
        decay_learning_rate:              bool = False
    ):

        assert n_regress_tasks == 0 and (type(n_classes) == int or len(n_classes) <= 1), "Only single-task classification RNN is supported now"

        new_run = run_num == -1
        super().__init__(layers, models_dir, log_key, n_regress_tasks, n_classes, task_names, config, run_num,
                         batch_size, record, random_state, data_params)

        param_names = ['n_timesteps', 'n_features', 'l2_lambda', 'learning_rate', 'beta1', 'beta2', 'decay_learning_rate']

        self.log_fname  = f"{models_dir}/log.h5"

        if type(n_features) == int:
            n_features = {'default': n_features}

        if new_run:
            assert n_timesteps != -1
            assert type(n_features) == dict
            self.n_timesteps         = n_timesteps
            self.n_features          = n_features
            self.l2_lambda           = l2_lambda
            self.learning_rate       = learning_rate
            self.beta1               = beta1
            self.beta2               = beta2
            self.decay_learning_rate = decay_learning_rate
        else:
            self.run_num = run_num
            log = pd.read_hdf(self.log_fname, log_key)
            for param in param_names:
                p = log.loc[run_num, param]
                self.__setattr__(param, p if type(p) != np.float64 else p.astype(np.float32))

        self.params.update({param: self.__getattribute__(param) for param in param_names})

        self._build_graph()

    def _batches_per_epoch(self, data) -> int:
        if type(data) == dict: # for GroupedLSTM
            n_batches = int(np.ceil(len(next(iter(data.values()))) / self.batch_size))
        else:
            n_batches = int(np.ceil(len(data) / self.batch_size))
        return n_batches

    @staticmethod
    def _metric_improved(old_metric: _numeric, new_metric: _numeric, significant: bool=False, threshold: float=.01) -> bool:
        if significant:
            return new_metric < (1 - threshold) * old_metric
        else:
            return new_metric < old_metric

    def _build_graph(self) -> None:
        """
        If self.log_dir contains a previously trained model, then the graph from that run is loaded for further
        training/inference. Otherwise, a new graph is built.
        Also starts a session with self.graph.
        If self.log_dir != '' then a Saver and summary writers are also created.
        """

        self.graph = tf.Graph()
        with self.graph.as_default():
            group_names = list(self.n_features.keys())
            self.inputs_p = {name: tf.placeholder(tf.float32, shape=(None, self.n_timesteps, self.n_features[name]),
                                                  name=f'inputs_p_{name}') for name in group_names}
            # TODO: repeat labels across n_timesteps instead of taking them in as that shape
            self.labels_p = tf.placeholder(tf.int32, shape=(None, 1), name='labels_p')
            self.is_training = tf.placeholder_with_default(False, [])

            labels = tf.tile(self.labels_p, [1, self.n_timesteps])
            data = tf.contrib.data.Dataset.from_tensor_slices(([self.inputs_p[name] for name in group_names], labels))
            self.data = data.batch(self.batch_size)

            iterator = tf.contrib.data.Iterator.from_structure(self.data.output_types, self.data.output_shapes)
            self.data_init_op = iterator.make_initializer(self.data)

            self.inputs, self.labels = iterator.get_next()

            hidden = self.inputs

            for layer in self.layers:
                hidden = layer.apply(hidden, is_training=self.is_training)

            self.logits = tf.layers.dense(hidden, self.n_classes[0], activation=None, name='logits')
            self.predict = tf.nn.softmax(self.logits, name='predict')
            # TODO: do we need all of the features to tell the length? If so, that will be an issue inside of the
            # separate LSTM modules. We should check this on the actual data and, if needed, find a better way to
            # tell length (e.g. some sentinel value that marks the end if 0 doesn't work for that)
            mask = tf.cast(tf.sequence_mask(LSTMLayer.length(self.inputs[0]), self.n_timesteps), tf.float32)
            self.loss_op = tf.contrib.seq2seq.sequence_loss(self.logits, self.labels, mask)

            # self.predict = tf.nn.sigmoid(self.logits, name='predict')
            # auc_val, self.auc_op = tf.contrib.metrics.streaming_auc(self.predict[:, 1], self.labels, name='auc_op')
            # self.loss_op = tf.losses.sparse_softmax_cross_entropy(self.labels, self.logits, scope='xent')

            if self.l2_lambda:
                self.loss_op = tf.add(self.loss_op, self.l2_lambda * tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()]), name='loss')

            # TODO: get these in use (so we can store epochs + time when reloading for further training)
            # self.epoch = tf.Variable(0, name='epoch', trainable=False)
            # self.train_time = tf.Variable(0, name='train_time', trainable=False)
            if self.decay_learning_rate:
                self.global_step = tf.Variable(0, name='global_step', trainable=False) # should be loaded like self.epoch
                self.decayed_lr = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                             decay_steps=30000 // self.batch_size, decay_rate=0.9)
                self.optimizer = tf.train.RMSPropOptimizer(self.decayed_lr)
                self.train_op = self.optimizer.minimize(self.loss_op, global_step=self.global_step, name='train_op')
            else:
                self.train_op = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2).minimize(self.loss_op, name='train_op')

            self.metrics = {}
            self.early_stop_metric_name = 'dev_loss'
            self.uses_dataset = True

            self.saver = tf.train.Saver()
            self.global_init = tf.global_variables_initializer()
            self.local_init = tf.local_variables_initializer()

        self._add_savers_and_writers()
        self._check_graph()
        #  TODO: make check_graph general enough to work (acc1 and acc5 need to be removed from BaseNN
        # need to implement a more flexible metrics like {'acc1': (tensor, 'how to combine when batching'), ...}

    def predict_proba(self, inputs, labels, final_only=False) -> pd.DataFrame:
        """
        Generates predictions (predicted 'probabilities', not binary labels) on the test set
        :param labels: these are also needed because the data iterator feeds in batches of inputs and labels.
                       Having the labels also lets us put the same index on the predictions.
        :returns: array of predicted probabilities of being positive for each sample in the test set
        """

        if self.n_classes > 2:
            assert False, "Only implemented for 2 classes so far. Otherwise, output is 3D (patients x timesteps x classes)"

        self.sess.run(self.data_init_op, self._get_feed_dict(inputs, labels))
        predictions = []
        for _ in range(self._batches_per_epoch(inputs)):
            predictions.append(self.sess.run(self.predict)[:, :, 1])
        predictions = np.concatenate(predictions)

        if final_only:
            assert False, "Need to implement this"

        return pd.DataFrame(predictions, index=labels.index)

    def score(self, inputs, labels, metric='auc'):
        if metric != 'auc':
            assert False, 'Only AUC supported so far'

        self.sess.run([self.data_init_op, self.local_init], self._get_feed_dict(inputs, labels))
        for _ in range(int(np.ceil(len(inputs) / self.batch_size))):
            auc = self.sess.run(self.auc_op)
        return auc


# def inception(
#         img_width:                        int = 128,
#         img_height:                       int = 128,
#         n_channels:                       int = 3,
#         n_classes:                        int = 2,
#         log_fname:                        str = f'{miniplaces}/models/log.h5',
#         log_key:                          str = 'default',
#         data_params: Optional[Dict[str, Any]] = None,
#         l2_lambda:            Optional[float] = None,
#         learning_rate:                  float = 0.001,
#         beta1:                          float = 0.9,
#         beta2:                          float = 0.999,
#         config:      Optional[tf.ConfigProto] = None,
#         run_num:                          int = -1,
#         batch_size:                       int = 64,
#         record:                          bool = True,
#         random_state:                     int = 521,):
#
#     inception_a = LayerModule([
#         BranchedLayer([AvgPoolLayer(1, 1), ConvLayer(96, 1), ConvLayer(64, 1), ConvLayer(64, 1)]),
#         BranchedLayer([ConvLayer(96, 1), None, ConvLayer(96, 3), ConvLayer(96, 3)]),
#         BranchedLayer([None, None, None, ConvLayer(96, 3)]),
#         MergeLayer(axis=3)
#     ])
#
#     inception_b = LayerModule([
#         BranchedLayer([AvgPoolLayer(1, 1), ConvLayer(384, 1), ConvLayer(192, 1), ConvLayer(192, 1)]),
#         BranchedLayer([ConvLayer(128, 1), None, ConvLayer(224, [7, 1]), ConvLayer(192, [1, 7])]),
#         BranchedLayer([None, None, ConvLayer(256, [1, 7]), ConvLayer(224, [7, 1])]),
#         BranchedLayer([None, None, None, ConvLayer(224, [1, 7])]),
#         BranchedLayer([None, None, None, ConvLayer(256, [7, 1])]),
#         MergeLayer(axis=3)
#     ])
#
#     inception_c = LayerModule([
#         BranchedLayer([AvgPoolLayer(1, 1), ConvLayer(256, 1), ConvLayer(384, 1), ConvLayer(384, 1)]),
#         BranchedLayer([ConvLayer(256, 1), None, BranchedLayer([ConvLayer(256, [1, 3]), ConvLayer(256, [3, 1])]), ConvLayer(448, [1, 3])]),
#         BranchedLayer([None, None, None, ConvLayer(512, [3, 1])]),
#         BranchedLayer([None, None, None, BranchedLayer([ConvLayer(256, [3, 1]), ConvLayer(256, [1, 3])])]),
#         MergeLayer(axis=3)
#     ])
#
#     layers = [
#         ConvLayer(32, 3, 2, padding='valid'),
#         ConvLayer(32, 3),
#         ConvLayer(64, 3),
#         BranchedLayer([MaxPoolLayer(3, padding='valid'), ConvLayer(96, 3, padding='valid')]), # don't use stride of 2 since our images are smaller
#         MergeLayer(axis=3),
#         BranchedLayer([ConvLayer(64, 1), ConvLayer(64, 1)]),
#         BranchedLayer([ConvLayer(96, 3, padding='valid'), ConvLayer(64, [7, 1])]),
#         BranchedLayer([None, ConvLayer(64, [1, 7])]),
#         BranchedLayer([None, ConvLayer(96, 3, padding='valid')]),
#         MergeLayer(axis=3),
#         BranchedLayer([ConvLayer(192, 3, strides=2, padding='valid'), MaxPoolLayer(3, strides=2, padding='valid')]),
#         MergeLayer(axis=3),
#         *([inception_a] * 1), # x4
#         ConvLayer(1024, 3, strides=2), # reduction_a
#         *([inception_b] * 1), # x7
#         ConvLayer(1536, 3, strides=2), # reduction_b
#         *([inception_c] * 1), # x3
#         GlobalAvgPoolLayer(),
#         FlattenLayer(),
#         DropoutLayer(rate=0.8)
#     ]
#
#     return CNN(layers=layers, img_width=img_width, img_height=img_height, n_channels=n_channels, n_classes=[n_classes],
#                log_fname=log_fname, log_key=log_key, data_params=data_params, l2_lambda=l2_lambda,
#                learning_rate=learning_rate, beta1=beta1, beta2=beta2, config=config, run_num=run_num,
#                batch_size=batch_size, record=record, random_state=random_state)

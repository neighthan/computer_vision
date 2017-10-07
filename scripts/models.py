from __future__ import print_function, division, absolute_import
import numpy as np
import pandas as pd
import tensorflow as tf
from tflearn import activations
from tqdm import tnrange, trange
import time
import os
import sys
from utils import tf_init, get_next_run_num, get_abs_path
from typing import List, Optional, Dict, Any
from layers import ConvLayer, MaxPoolLayer, AvgPoolLayer, BranchedLayer, MergeLayer, LayerModule, FlattenLayer,\
    DenseLayer, DropoutLayer, _Layer
import warnings

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

miniplaces = get_abs_path('../miniplaces/')


class BaseNN(object):
    """
    This class implements several methods that may be used by neural networks in general. It doesn't actually create any
    layers, so it shouldn't be used directly.
    Currently, this class only works for classification, not regression. It also defines the score method already and expects that
    the model can calculate accuracy@1 and accuracy@5, so it expects >= 5-way classification.
    """

    def __init__(
        self,
        log_fname        = '{}/models/log.h5'.format(miniplaces),
        log_key          = 'default',
        config           = None,
        run_num          = -1,
        batch_size       = 128,
        record           = True,
        random_state     = 521
    ):
        # don't include run_num or config
        param_names = ['random_state', 'batch_size', 'record']

        if config is None:
            config = tf_init()

        if run_num == -1:
            self.run_num      = get_next_run_num('{}/models/run_num.pkl'.format(miniplaces))
            self.random_state = random_state
            self.batch_size   = batch_size
            self.record       = record
        else:
            self.run_num = run_num
            log = pd.read_hdf(log_fname, log_key)
            for param in param_names:
                p = log.loc[run_num, param]
                self.__setattr__(param, p if type(p) != np.float64 else p.astype(np.float32))

        self.config    = config
        self.log_fname = log_fname
        self.log_key   = log_key
        self.params    = {param: self.__getattribute__(param) for param in param_names}

        if record:
            self.log_dir = '{}/models/{}/'.format(miniplaces, self.run_num)
            try:
                os.makedirs(self.log_dir)
            except OSError: # already exists if loading saved model
                pass
        else:
            self.log_dir = ''

        tf.set_random_seed(self.random_state)
        np.random.seed(self.random_state)

    def __check_graph__(self):
        required_attrs = ['loss_op', 'train_op', 'inputs_p', 'labels_p', 'probs_p', 'accuracy1', 'accuracy5', 'predict',
                          'is_training']
        for attr in required_attrs:
            getattr(self, attr)

    def __build_graph__(self):
        """
        This method should be overridden by all subclasses. The code below is just starter code.
        """

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            self.global_init = tf.global_variables_initializer()
            self.is_training = tf.placeholder_with_default(False, [])

            self.probs_p = tf.placeholder(tf.float32, shape=(None, self.n_classes), name='probs_p')
            self.accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.probs_p, self.labels_p, k=1), tf.float32))
            self.accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.probs_p, self.labels_p, k=5), tf.float32))

        self.__add_savers_and_writers__()
        self.__check_graph__()
    
    def __add_savers_and_writers__(self):
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
            print("Loading graph from:", meta_graph_file)
            saver = tf.train.import_meta_graph(meta_graph_file)
            saver.restore(self.sess, os.path.join(self.log_dir, 'model.ckpt'))
        else:
            self.sess.run(self.global_init)
    
    def predict_proba(self, inputs):
        """
        Generates predictions (predicted 'probabilities', not binary labels)
        :returns: array of predicted probabilities of being positive for each sample in the test set
        :rtype: pd.DataFrame
        """

        predictions = []
        idx = list(range(len(inputs)))
        for batch in range(int(np.ceil(len(inputs) / self.batch_size))):
            batch_idx = idx[batch * self.batch_size : (batch + 1) * self.batch_size]
            predictions.append(self.sess.run(self.predict, {self.inputs_p: inputs[batch_idx]}))
        return pd.DataFrame(np.concatenate(predictions))

    def __save__(self):
        self.saver.save(self.sess, os.path.join(self.log_dir, 'model.ckpt'))

    def __add_summaries__(self, epoch, train_loss, dev_loss, dev_acc1, dev_acc5):
        summary_str = self.sess.run(self.summary_op)
        self.train_writer.add_summary(summary_str, epoch)

        self.train_writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='loss', simple_value=train_loss)]), epoch)
        self.dev_writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='loss', simple_value=dev_loss)]), epoch)
        self.dev_writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='acc1', simple_value=dev_acc1)]), epoch)
        self.dev_writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='acc5', simple_value=dev_acc5)]), epoch)

    def train(self, train_inputs, train_labels, dev_inputs, dev_labels, n_epochs=100, max_patience=5, in_notebook=False):
        """
        The best epoch is the one where the accuracy on the dev set is the highest. "best" in reference to other metrics
        (e.g. dev accuracy@5) means the value of that metric at the best epoch.
        :param train_inputs:
        :param train_labels:
        :param dev_inputs:
        :param dev_labels:
        :param n_epochs:
        :param max_patience:
        :param in_notebook:
        :returns: best_dev_acc1, best_dev_acc5, best_train_loss, best_dev_loss, train_time, run_num
        """

        start_time = time.time()

        if in_notebook:
            epoch_range = lambda *args: tnrange(*args, unit='epoch')
            batch_range = lambda *args: tnrange(*args, unit='batch', leave=False)
        else:
            epoch_range = lambda *args: trange(*args, unit='epoch')
            batch_range = lambda *args: trange(*args, unit='batch', leave=False)

        best_train_loss = best_dev_loss = np.inf
        best_dev_acc1 = best_dev_acc5 = early_stop_acc = 0
        patience = max_patience
        train_batches_per_epoch = int(np.ceil(len(train_inputs) / self.batch_size))
        dev_batches_per_epoch   = int(np.ceil(len(dev_inputs) / self.batch_size))
        
        train_idx = list(range(len(train_labels)))
        dev_idx = list(range(len(dev_labels)))

        epochs = epoch_range(n_epochs)
        for epoch in epochs:
            np.random.shuffle(train_idx)

            train_loss = []
            batches = batch_range(train_batches_per_epoch)
            for batch in batches:
                batch_idx = train_idx[batch * self.batch_size : (batch + 1) * self.batch_size]
                loss, _ = self.sess.run([self.loss_op, self.train_op], {self.inputs_p: train_inputs[batch_idx],
                                                                        self.labels_p: train_labels[batch_idx],
                                                                        self.is_training: True})
                train_loss.append(loss)
                batches.set_description('{:.3f}'.format(loss))

            dev_loss = []
            preds = []
            batches = batch_range(dev_batches_per_epoch)
            for batch in batches:
                batch_idx = dev_idx[batch * self.batch_size : (batch + 1) * self.batch_size]
                loss, pred = self.sess.run([self.loss_op, self.predict], {self.inputs_p: dev_inputs[batch_idx],
                                                            self.labels_p: dev_labels[batch_idx]})
                dev_loss.append(loss)
                preds.append(pred)
                batches.set_description('{:.3f}'.format(loss))

            train_loss = np.mean(train_loss)
            dev_loss = np.mean(dev_loss)
            preds = np.concatenate(preds)
            dev_acc1, dev_acc5 = self.sess.run([self.accuracy1, self.accuracy5], {self.probs_p: preds, self.labels_p: dev_labels})

            if self.record:
                self.__add_summaries__(epoch, train_loss, dev_loss, dev_acc1, dev_acc5)

            if dev_acc5 > best_dev_acc5: # always keep updating the best model
                best_dev_acc1 = dev_acc1
                best_dev_acc5 = dev_acc5
                best_dev_loss = dev_loss
                best_train_loss = train_loss
                train_time = (time.time() - start_time) / 60 # in minutes
                if self.record:
                    self.__log__({'train_loss': best_train_loss, 'dev_loss': best_dev_loss, 'train_time': train_time,
                                  'train_complete': False})
                    self.__save__()

            if dev_acc5 > 1.01 * early_stop_acc:
                # keeping this separate means the model has max_patience epochs to increase >1%
                # instead of potentially just having one epoch (if we used best_dev_auc, which is
                # continually updated)
                patience = max_patience
                early_stop_acc = dev_acc5
            else:
                patience -= 1
                if patience == 0:
                    break

            runtime = int((time.time() - start_time) / 60)
            epochs.set_description("Epoch {}. Train Loss: {:.3f}. Dev loss: {:.3f}. Runtime {}."\
                  .format(epoch + 1, best_train_loss, best_dev_loss, runtime))

        train_time = (time.time() - start_time) / 60 # in minutes
        if self.record:
            self.__log__({'train_loss': best_train_loss, 'dev_loss': best_dev_loss, 'train_time': train_time,
                         'train_complete': True})
            self.saver.restore(self.sess, os.path.join(self.log_dir, 'model.ckpt')) # reload best epoch

        return best_dev_acc1, best_dev_acc5, best_train_loss, best_dev_loss, train_time, self.run_num

    def __log__(self, extras):
        params_df = (pd.DataFrame([[self.params[key] for key in self.params.keys()]], columns=self.params.keys(), index=[self.run_num])
                     .assign(**extras))

        try:
            log = pd.read_hdf(self.log_fname, self.log_key)
            
            if self.run_num in log.index:
                log = log.drop(self.run_num)
            pd.concat([log, params_df]).to_hdf(self.log_fname, self.log_key)
        except (IOError, KeyError):  # this file or key doesn't exist yet
            params_df.to_hdf(self.log_fname, self.log_key)

    def score(self, inputs, labels):
        probs = self.predict_proba(inputs)
        return self.sess.run([self.accuracy1, self.accuracy5], {self.probs_p: probs, self.labels_p: labels})


class PretrainedCNN(BaseNN):
    """
    """
    def __init__(
        self,
        img_width        = 128,
        img_height       = 128,
        n_channels       = 3,
        n_classes        = None,
        log_fname        = '{}/models/log.h5'.format(miniplaces),
        log_key          = 'default',
        data_params      = None,
        dense_nodes      = (128,),
        l2_lambda        = None,
        learning_rate    = 0.001,
        beta1            = 0.9,
        beta2            = 0.999,
        config           = None,
        run_num          = -1,
        batch_size       = 64,
        record           = True,
        random_state     = 521,
        dense_activation = 'relu',
        finetune         = False,
        pretrained_weights = False,
        cnn_module       = 'vgg16'
    ):
        new_run = run_num == -1
        super(PretrainedCNN, self).__init__(log_fname, log_key, config, run_num, batch_size, record, random_state)

        param_names = ['img_width', 'img_height', 'n_channels', 'n_classes', 'dense_nodes', 'l2_lambda', 'learning_rate',
                       'beta1', 'beta2', 'dense_activation', 'finetune', 'cnn_module', 'pretrained_weights']

        if not pretrained_weights:
            finetune = True

        if data_params is None:
            data_params = {}

        if new_run:
            assert type(n_classes) is not None
            self.img_height = img_height
            self.img_width = img_width
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.dense_nodes = dense_nodes
            self.l2_lambda = l2_lambda
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.dense_activation = dense_activation
            self.finetune = finetune
            self.cnn_module = cnn_module
            self.pretrained_weights = pretrained_weights
        else:
            log = pd.read_hdf(log_fname, log_key)
            for param in param_names:
                p = log.loc[run_num, param]
                self.__setattr__(param, p if type(p) != np.float64 else p.astype(np.float32))

        self.params.update(data_params)
        self.params.update({param: self.__getattribute__(param) for param in param_names})
        
        self.__build_graph__()

    def __build_graph__(self):
        """
        If self.log_dir contains a previously trained model, then the graph from that run is loaded for further
        training/inference. Otherwise, a new graph is built.
        Also starts a session with self.graph.
        If self.log_dir != '' then a Saver and summary writers are also created.
        :returns: None
        """

        self.graph = tf.Graph()
        self.sess = tf.Session(config=self.config, graph=self.graph)

        with self.graph.as_default(), self.sess.as_default():
            self.inputs_p = tf.placeholder(tf.float32, shape=(None, self.img_height, self.img_width, self.n_channels), name='inputs_p')
            self.labels_p = tf.placeholder(tf.int32, shape=None, name='labels_p')
            self.is_training = tf.placeholder_with_default(False, [])
            
            with tf.variable_scope('cnn'):
                weights = 'imagenet' if self.pretrained_weights else None

                cnn = cnn_modules[self.cnn_module]
                cnn_out = cnn(include_top=False, input_tensor=self.inputs_p, weights=weights).output
                hidden = tf.contrib.layers.flatten(cnn_out)

            dense_activation = activations[self.dense_activation]
            with tf.variable_scope('dense'):
                for i in range(len(self.dense_nodes)):
                    hidden = tf.layers.dense(hidden, self.dense_nodes[i], activation=dense_activation, name='dense_{}'.format(i))
                self.logits = tf.layers.dense(hidden, self.n_classes, activation=None, name='logits')

            self.predict = tf.nn.softmax(self.logits, name='predict')
            self.loss_op = tf.losses.sparse_softmax_cross_entropy(self.labels_p, self.logits, scope='xent')

            if self.l2_lambda:
                self.loss_op = tf.add(self.loss_op, self.l2_lambda * tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()]), name='loss')

            trainable_vars = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            if not self.finetune: # don't finetune CNN layers
                trainable_vars = filter(lambda tensor: not tensor.name.startswith('cnn'), trainable_vars)
            
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2)
            self.train_op = self.optimizer.minimize(self.loss_op, var_list=trainable_vars, name='train_op')

            self.probs_p = tf.placeholder(tf.float32, shape=(None, self.n_classes), name='probs_p')
            self.accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.probs_p, self.labels_p, k=1), tf.float32))
            self.accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.probs_p, self.labels_p, k=5), tf.float32))

            self.saver = tf.train.Saver()
            self.global_init = tf.global_variables_initializer()
        self.__add_savers_and_writers__()
        self.__check_graph__()


class CNN(BaseNN):
    """
    """
    def __init__(
        self,
        layers:        Optional[List[_Layer]] = None,
        img_width:                        int = 128,
        img_height:                       int = 128,
        n_channels:                       int = 3,
        n_classes:                        int = 2,
        log_fname:                        str = '{}/models/log.h5'.format(miniplaces),
        log_key:                          str = 'default',
        data_params: Optional[Dict[str, Any]] = None,
        l2_lambda:            Optional[float] = None,
        learning_rate:                  float = 0.001,
        beta1:                          float = 0.9,
        beta2:                          float = 0.999,
        config:      Optional[tf.ConfigProto] = None,
        run_num:                          int = -1,
        batch_size:                       int = 64,
        record:                          bool = True,
        random_state:                     int = 521,
    ):
        new_run = run_num == -1
        super(CNN, self).__init__(log_fname, log_key, config, run_num, batch_size, record, random_state)

        param_names = ['img_width', 'img_height', 'n_channels', 'n_classes', 'l2_lambda', 'learning_rate',
                       'beta1', 'beta2', 'random_state', 'batch_size', 'layers']

        if data_params is None:
            data_params = {}

        if new_run:
            assert type(layers) is not None
            self.layers = layers
            self.img_height = img_height
            self.img_width = img_width
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.l2_lambda = l2_lambda
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
        else:
            log = pd.read_hdf(log_fname, log_key)
            for param in param_names:
                p = log.loc[run_num, param]
                self.__setattr__(param, p if type(p) != np.float64 else p.astype(np.float32))

        self.params.update(data_params)
        self.params.update({param: self.__getattribute__(param) for param in param_names})

        self.__build_graph__()


    def __build_graph__(self):
        """
        If self.log_dir contains a previously trained model, then the graph from that run is loaded for further
        training/inference. Otherwise, a new graph is built.
        Also starts a session with self.graph.
        If self.log_dir != '' then a Saver and summary writers are also created.
        :returns: None
        """

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs_p = tf.placeholder(tf.float32, shape=(None, self.img_height, self.img_width, self.n_channels), name='inputs_p')
            self.labels_p = tf.placeholder(tf.int32, shape=None, name='labels_p')
            self.is_training = tf.placeholder_with_default(False, [])

            hidden = self.inputs_p

            for layer in self.layers:
                hidden = layer.apply(hidden, is_training=self.is_training)

            self.logits = tf.layers.dense(hidden, self.n_classes, activation=None, name='logits')

            self.predict = tf.nn.softmax(self.logits, name='predict')
            self.loss_op = tf.losses.sparse_softmax_cross_entropy(self.labels_p, self.logits, scope='xent')

            if self.l2_lambda:
                self.loss_op = tf.add(self.loss_op, self.l2_lambda * tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()]), name='loss')

            self.train_op = tf.train.AdamOptimizer(self.learning_rate, self.beta1, self.beta2).minimize(self.loss_op, name='train_op')

            self.probs_p = tf.placeholder(tf.float32, shape=(None, self.n_classes), name='probs_p')
            self.accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.probs_p, self.labels_p, k=1), tf.float32))
            self.accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.probs_p, self.labels_p, k=5), tf.float32))

            self.saver = tf.train.Saver()
            self.global_init = tf.global_variables_initializer()

        self.__add_savers_and_writers__()
        self.__check_graph__()

def inception(
        img_width:                        int = 128,
        img_height:                       int = 128,
        n_channels:                       int = 3,
        n_classes:                        int = 2,
        log_fname:                        str = '{}/models/log.h5'.format(miniplaces),
        log_key:                          str = 'default',
        data_params: Optional[Dict[str, Any]] = None,
        l2_lambda:            Optional[float] = None,
        learning_rate:                  float = 0.001,
        beta1:                          float = 0.9,
        beta2:                          float = 0.999,
        config:      Optional[tf.ConfigProto] = None,
        run_num:                          int = -1,
        batch_size:                       int = 64,
        record:                          bool = True,
        random_state:                     int = 521,):

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
        BranchedLayer([MaxPoolLayer(3, padding='valid'), ConvLayer(96, 3, padding='valid')]), # don't use stride of 2 since our images are smaller
        MergeLayer(axis=3),
        BranchedLayer([ConvLayer(64, 1), ConvLayer(64, 1)]),
        BranchedLayer([ConvLayer(96, 3, padding='valid'), ConvLayer(64, [7, 1])]),
        BranchedLayer([None, ConvLayer(64, [1, 7])]),
        BranchedLayer([None, ConvLayer(96, 3, padding='valid')]),
        MergeLayer(axis=3),
        BranchedLayer([ConvLayer(192, 3, strides=2, padding='valid'), MaxPoolLayer(3, strides=2, padding='valid')]),
        MergeLayer(axis=3),
        *([inception_a] * 4),
        ConvLayer(1024, 3, strides=2), # reduction_a
        *([inception_b] * 7),
        ConvLayer(1536, 3, strides=2), # reduction_b
        *([inception_c] * 3),
        AvgPoolLayer(8, 1, padding='valid'),
        FlattenLayer(),
        DropoutLayer(rate=0.8)
    ]

    return CNN(layers=layers, img_width=img_width, img_height=img_height, n_channels=n_channels, n_classes=n_classes,
               log_fname=log_fname, log_key=log_key, data_params=data_params, l2_lambda=l2_lambda,
               learning_rate=learning_rate, beta1=beta1, beta2=beta2, config=config, run_num=run_num,
               batch_size=batch_size, record=record, random_state=random_state)

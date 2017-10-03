from __future__ import print_function, division, absolute_import
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tnrange
import time
import os
import sys
from utils import tf_init, get_next_run_num, get_abs_path

cnn_modules = {
    'vgg16': tf.contrib.keras.applications.VGG16,
    'xception': tf.contrib.keras.applications.Xception
}

miniplaces = get_abs_path('../miniplaces/')


class PretrainedCNN(object):
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
        dense_activation = tf.nn.relu,
        finetune         = False,
        cnn_module       = 'vgg16'
    ):
        param_names = ['img_width', 'img_height', 'n_channels', 'n_classes', 'dense_nodes', 'l2_lambda', 'learning_rate',
                       'beta1', 'beta2', 'random_state', 'batch_size', 'dense_activation', 'finetune', 'cnn_module']

        if config is None:
            config = tf_init()
        if data_params is None:
            data_params = {}

        if run_num == -1:
            self.run_num = get_next_run_num('{}/models/run_num.pkl'.format(miniplaces))
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
            self.random_state = random_state
            self.batch_size = batch_size
            self.dense_activation = dense_activation
            self.finetune = finetune
            self.cnn_module = cnn_module
        else:
            self.run_num = run_num
            log = pd.read_hdf(log_fname, log_key)
            for param in param_names:
                p = log.loc[run_num, param]
                self.__setattr__(param, p if type(p) != np.float64 else p.astype(np.float32))

        self.config    = config
        self.record    = record
        self.log_fname = log_fname
        self.log_key   = log_key
        self.params    = data_params.copy()
        self.params.update({param: self.__getattribute__(param) for param in param_names})

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

        self.__build_graph__()


    def __build_graph__(self):
        """
        If self.log_dir contains a previously trained model, then the graph from that run is loaded for further
        training/inference. Otherwise, a new graph is built.
        Also starts a session with self.graph.
        If self.log_dir != '' then a Saver and summary writers are also created.
        :returns: None
        """

        meta_graph_file = os.path.join(self.log_dir, 'model.ckpt.meta')
        self.graph = tf.Graph()
        self.sess = tf.Session(config=self.config, graph=self.graph)

        with self.graph.as_default(), self.sess.as_default():
            self.inputs_p = tf.placeholder(tf.float32, shape=(None, self.img_height, self.img_width, self.n_channels), name='inputs_p')
            self.labels_p = tf.placeholder(tf.int32, shape=None, name='labels_p')
            self.onehot_labels = tf.one_hot(self.labels_p, depth=self.n_classes, name='onehot_labels')

#             data = tf.contrib.data.Dataset.from_tensor_slices((self.inputs_p, self.labels_p))
            
#             self.data = data.batch(self.batch_size)

#             iterator = tf.contrib.data.Iterator.from_structure(self.data.output_types, self.data.output_shapes)
#             self.data_init_op = iterator.make_initializer(self.data)

#             self.inputs, self.labels = iterator.get_next()
            
            with tf.variable_scope('cnn'):
                cnn = cnn_modules[self.cnn_module]
                cnn_out = cnn(include_top=False, input_tensor=self.inputs_p).output
                hidden = tf.contrib.layers.flatten(cnn_out)
            
            with tf.variable_scope('dense'):
                for i in range(len(self.dense_nodes)):
                    hidden = tf.layers.dense(hidden, self.dense_nodes[i], activation=self.dense_activation, name='dense_{}'.format(i))
                self.logits = tf.layers.dense(hidden, self.n_classes, activation=None, name='logits')

            self.predict = tf.nn.softmax(self.logits, name='predict')
            self.loss_op = tf.losses.softmax_cross_entropy(self.onehot_labels, self.logits, scope='xent')

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

        if self.log_dir != '':
            with self.graph.as_default():
                self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'train'), self.graph,
                                                          flush_secs=30)
                self.dev_writer = tf.summary.FileWriter(os.path.join(self.log_dir, 'dev'), self.graph,
                                                        flush_secs=30)
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.name, var)
                self.summary_op = tf.summary.merge_all()

        if os.path.isfile(meta_graph_file):  # load saved model
            print("Loading graph from:", meta_graph_file)
            saver = tf.train.import_meta_graph(meta_graph_file)
            saver.restore(self.sess, os.path.join(self.log_dir, 'model.ckpt'))
        else:
            self.sess.run(self.global_init)

    def predict_proba(self, inputs):
        """
        Generates predictions (predicted 'probabilities', not binary labels) on the test set
        :param labels: these are also needed because the data iterator feeds in batches of inputs and labels.
                       Having the labels also lets us put the same index on the predictions.
        :returns: array of predicted probabilities of being positive for each sample in the test set
        :rtype: pd.DataFrame
        """

#         self.sess.run(self.data_init_op, {self.inputs_p: inputs.values, self.labels_p: labels.values})

        predictions = []
        idx = list(range(len(inputs)))
        for batch in range(int(np.ceil(len(inputs) / self.batch_size))):
            batch_idx = idx[batch * self.batch_size : (batch + 1) * self.batch_size]
            predictions.append(self.sess.run(self.predict, {self.inputs_p: inputs[batch_idx]}))
        return pd.DataFrame(np.concatenate(predictions))

    def __save__(self):
        self.saver.save(self.sess, os.path.join(self.log_dir, 'model.ckpt'))

    def __add_summaries__(self, epoch, train_loss, dev_loss):
        summary_str = self.sess.run(self.summary_op)
        self.train_writer.add_summary(summary_str, epoch)

        self.train_writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='loss', simple_value=train_loss)]), epoch)
        self.dev_writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag='loss', simple_value=dev_loss)]), epoch)

    def train(self, train_inputs, train_labels, dev_inputs, dev_labels, n_epochs=100, max_patience=5, in_notebook=False,
             verbose=False):
        start_time = time.time()
        
        train_labels = train_labels.astype(np.int32)
        dev_labels = dev_labels.astype(np.int32)

        if in_notebook:
            epoch_range = lambda *args: tnrange(*args, unit='epoch')
            batch_range = lambda *args: tnrange(*args, unit='batch', leave=False)            
        else:
            epoch_range = range
            batch_range = range

        early_stop_loss = np.inf
        best_dev_loss = np.inf
        best_epoch = 0
        patience = max_patience
        train_batches_per_epoch = int(np.ceil(len(train_inputs) / self.batch_size))
        dev_batches_per_epoch   = int(np.ceil(len(dev_inputs) / self.batch_size))
        
        train_idx = list(range(len(train_labels)))
        dev_idx = list(range(len(dev_labels)))

        for epoch in epoch_range(n_epochs):
            np.random.shuffle(train_idx)
#             self.sess.run(self.data_init_op, {self.inputs_p: train_inputs, self.labels_p: train_labels})
            train_loss = []
            for batch in batch_range(train_batches_per_epoch):
                batch_idx = train_idx[batch * self.batch_size : (batch + 1) * self.batch_size]
                loss, _ = self.sess.run([self.loss_op, self.train_op], {self.inputs_p: train_inputs[batch_idx],
                                                                       self.labels_p: train_labels[batch_idx]})
                train_loss.append(loss)
            
#             self.sess.run(self.data_init_op, {self.inputs_p: dev_inputs, self.labels_p: dev_labels})
            dev_loss = []
            for batch in batch_range(dev_batches_per_epoch):
                batch_idx = dev_idx[batch * self.batch_size : (batch + 1) * self.batch_size]
                dev_loss.append(self.sess.run(self.loss_op, {self.inputs_p: dev_inputs[batch_idx],
                                                            self.labels_p: dev_labels[batch_idx]}))

            train_loss = np.mean(train_loss)
            dev_loss = np.mean(dev_loss)

            if self.record:
                self.__add_summaries__(epoch, train_loss, dev_loss)

            if dev_loss < best_dev_loss: # always keep updating the best model
                best_dev_loss = dev_loss
                best_train_loss = train_loss
                best_epoch = epoch
                train_time = (time.time() - start_time) / 60 # in minutes
                if self.record:
                    self.__log__({'train_loss': best_train_loss, 'dev_loss': best_dev_loss, 'train_time': train_time,
                                  'train_complete': False})
                    self.__save__()

            if dev_loss < .99 * early_stop_loss:
                # keeping this separate means the model has max_patience epochs to increase >1%
                # instead of potentially just having one epoch (if we used best_dev_auc, which is
                # continually updated)
                patience = max_patience
                early_stop_loss = dev_loss
            else:
                patience -= 1
                if patience == 0:
                    print("Early stop! Saved model is epoch {}, which had loss = {:.3f}\
                    (run number {})".format(best_epoch + 1, best_dev_loss, self.run_num))
                    break
            runtime = int((time.time() - start_time) / 60)
            if verbose:
                print("Epoch {}. Train Loss: {:.3f}. Dev loss: {:.3f}. Runtime {}."\
                  .format(epoch + 1, best_train_loss, best_dev_loss, runtime))

        train_time = (time.time() - start_time) / 60 # in minutes
        if self.record:
            self.__log__({'train_loss': best_train_loss, 'dev_loss': best_dev_loss, 'train_time': train_time,
                         'train_complete': True})
            self.saver.restore(self.sess, os.path.join(self.log_dir, 'model.ckpt')) # reload best epoch

        return best_train_loss, best_dev_loss, train_time, self.run_num

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
#         self.sess.run([self.data_init_op], {self.inputs_p: inputs, self.labels_p: labels})
        probs = self.predict_proba(inputs)
        return self.sess.run([self.accuracy1, self.accuracy5], {self.probs_p: probs, self.labels_p: labels})

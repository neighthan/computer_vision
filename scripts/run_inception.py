#! /usr/bin/env python

import numpy as np
import tensorflow as tf
from utils import tf_init, load_data
from models import inception

config = tf_init()
tf.logging.set_verbosity(tf.logging.WARN)

train_inputs, train_labels, val_inputs, val_labels, test_inputs = load_data('miniplaces')
n_classes = len(np.unique(train_labels))


cnn = inception(n_classes=n_classes)# l2_lambda=.001)
cnn.train(train_inputs, train_labels, val_inputs, val_labels)

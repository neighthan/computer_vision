import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
import sys
from utils import tf_init, get_next_run_num, load_data, output_file
from layers import ConvLayer, MaxPoolLayer, AvgPoolLayer, BranchedLayer, MergeLayer, LayerModule, FlattenLayer, DenseLayer, GlobalAvgPoolLayer, DropoutLayer
from models import CNN
from argparse import ArgumentParser


def log(string):
    with open(fname, 'a+') as f:
        f.write(string + '\n')


parser = ArgumentParser()
parser.add_argument('-lr', '--learning_rate', type=float)
parser.add_argument('-l2', '--l2_lambda', type=float, default=None)
parser.add_argument('-d', '--device', type=int)

args = parser.parse_args()
l2_lambda = args.l2_lambda
learning_rate = args.learning_rate

config = tf_init(args.device)

train_inputs, train_labels, val_inputs, val_labels, test_inputs = load_data('miniplaces')
n_classes = len(np.unique(train_labels))
fname = os.path.expanduser(f'~/logs/l2_{l2_lambda}_lr_{learning_rate}')

labels = tf.placeholder(tf.int32, shape=None)
img = tf.placeholder(tf.float32, (None, 128, 128, 3))
drop_prob = tf.placeholder_with_default(0.0, shape=())

# output is 2x2x2048 for our images
cnn = tf.contrib.keras.applications.InceptionV3(include_top=False, weights=None, input_tensor=img, pooling=None)

layers = [
    tf.layers.Conv2D(2048, 2, padding='SAME', activation=tf.nn.relu),
    tf.layers.AveragePooling2D(2, 1, padding='valid'),
    tf.layers.Flatten(),
    tf.layers.Dropout(drop_prob)
]

hidden = cnn.output
for layer in layers:
    hidden = layer(hidden)

logits = tf.layers.Dense(n_classes, activation=None)(hidden)
preds = tf.nn.softmax(logits)
loss_op = tf.losses.sparse_softmax_cross_entropy(labels, logits)

if l2_lambda:
    loss_op = tf.add(loss_op, l2_lambda * tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()]), name='loss')

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

_, acc_op = tf.metrics.accuracy(labels, tf.argmax(preds, axis=1))

global_init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()

batch_size = 128

train_idx = list(range(len(train_labels)))
val_idx = list(range(len(val_labels)))

sess = tf.Session(config=config)
sess.run(global_init)

n_epochs = 100

epoch_frac = 5

for epoch in range(n_epochs):
    np.random.shuffle(train_idx)

    sess.run(local_init)
    train_loss = []
    tf.keras.backend.set_learning_phase(1)
    for batch in range(int(np.ceil(len(train_labels) / batch_size)) // epoch_frac):
        batch_idx = train_idx[batch * batch_size : (batch + 1) * batch_size]
        loss, train_acc, _ = sess.run([loss_op, acc_op, train_op], {img: train_inputs[batch_idx], labels: train_labels[batch_idx],
                                                                   drop_prob: 0.5, 'batch_normalization/keras_learning_phase:0': True})
        train_loss.append(loss)

    sess.run(local_init)
    val_loss = []
    tf.keras.backend.set_learning_phase(0)
    for batch in range(int(np.ceil(len(val_labels) / batch_size))):
        batch_idx = val_idx[batch * batch_size : (batch + 1) * batch_size]
        loss, val_acc = sess.run([loss_op, acc_op], {img: val_inputs[batch_idx], labels: val_labels[batch_idx],
                                                     'batch_normalization/keras_learning_phase:0': False})
        val_loss.append(loss)

    log(f"Epoch {epoch}. Train Loss: {np.mean(train_loss):.3f}; Val Loss: {np.mean(val_loss):.3f}. Train Acc: {train_acc:.3f}; Val Acc: {val_acc:.3f}")

    if np.mean(val_loss) > 100:
        break

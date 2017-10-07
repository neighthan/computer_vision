#! /usr/bin/env python

import numpy as np
from utils import tf_init, load_data
from layers import ConvLayer, MaxPoolLayer, AvgPoolLayer, BranchedLayer, MergeLayer, LayerModule, FlattenLayer, DenseLayer
from models import CNN

config = tf_init()

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
    FlattenLayer()
]

cnn = CNN(n_classes=n_classes, batch_size=64, layers=layers)# l2_lambda=.001)
cnn.train(train_inputs, train_labels, val_inputs, val_labels)

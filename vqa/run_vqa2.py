import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle
import json
from tf_layers.layers import EmbeddingLayer, ConvLayer, FlattenLayer, LSTMLayer, DenseLayer,\
    BranchedLayer, MergeLayer, CustomLayer, LayerModule, GlobalAvgPoolLayer, GlobalMaxPoolLayer,\
    add_implied_layers, DropoutLayer
from tf_layers.models import NN, get_inputs_from_spec
from tf_layers.tf_utils import tf_init
from computer_vision.vqa.vqa_utils import load_data, generator

config = tf_init()

train_inputs, train_labels, val_inputs, val_labels = load_data(['train', 'val'], use_glove=True, embed_object_names=False, return_img_paths=False)

del train_inputs['objects']
del val_inputs['objects']

input_spec = {
    'question': (train_inputs['question'].shape[1:], 'float32')
}

layers = [
    # question: (batches x question_length x glove_n_features)
    LSTMLayer(256, scope='lstm_question'),
    DenseLayer(512, batch_norm=''),
    DropoutLayer(0.75)
]

model_name = 'qonly__glove__ml2_1__dp_0.75__lstm_256__dense_512'

model = NN(input_spec, layers, n_classes=len(np.unique(train_labels['default'])), config=config,
           model_name=model_name, models_dir='models', modified_l2=True, l2_lambda=1,
           overwrite_saved=True
           )

model.train(train_inputs, train_labels, val_inputs, val_labels, verbose=1)

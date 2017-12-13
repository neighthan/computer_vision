import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle

from tf_layers.models import NN
from tf_layers.tf_utils import tf_init
from computer_vision.vqa.vqa_utils import load_data, get_layers, generator

config = tf_init()

n_glances = 5
include_img = True
batch_size = 64
question_lstm_size = 256
l2_lambda = 1

train_inputs, train_labels, val_inputs, val_labels = load_data(['train', 'val'])
mean = np.load('data/train/mean.npy')
std = np.load('data/train/std.npy')

n_object_types = len(np.unique(train_inputs['object_names']))
max_n_objects = train_inputs['objects'].shape[1]

input_spec = {
    'question': (train_inputs['question'].shape[1:], 'float32'),
    'objects': (train_inputs['objects'].shape[1:], 'float32'),
    'object_names': (train_inputs['object_names'].shape[1:], 'int32'),
}

if include_img:
    img_features_shape = np.load('/cluster/nhunt/img_features/train/features0.npy').shape[1:]
    input_spec['img'] = (img_features_shape, 'float32')

train_generator = lambda : generator(train_inputs, batch_size, 'train', labels=train_labels, mean=mean, std=std)
val_generator = lambda : generator(val_inputs, batch_size, 'val', labels=val_labels, mean=mean, std=std)

layers = get_layers(n_object_types, max_n_objects, n_glances=n_glances,
                    include_img=include_img,
                    question_lstm_size=question_lstm_size,
                    input_fusion=True,
                    use_bilinear_pool=True
                    )

model_name = f'rn50f__g2__bp__qlstm_{question_lstm_size}__dp_0.7'
if l2_lambda:
    model_name += f"__ml2_{l2_lambda}"

model = NN(input_spec, layers, n_classes=len(np.unique(train_labels['default'])), config=config,
            model_name=model_name, models_dir='models', l2_lambda=l2_lambda, batch_size=batch_size,
            modified_l2=True,
            # record=False,
            overwrite_saved=True
           )

model.train(train_generator=train_generator, dev_generator=val_generator, verbose=1,
            n_train_batches_per_epoch=250, n_dev_batches_per_epoch=100,
            n_epochs=100, max_patience=20)

model.update_tensor('learning_rate', .0001)

model.train(train_generator=train_generator, dev_generator=val_generator, verbose=1,
            n_train_batches_per_epoch=125, n_dev_batches_per_epoch=100,
            n_epochs=100, max_patience=20)

from argparse import ArgumentParser
import os
import numpy as np
import tensorflow as tf
from utils import load_data, get_abs_path
from gpu_dashboard.tf_utils import tf_init
from models import CNN
from layers import BranchedLayer, AvgPoolLayer, MergeLayer, LayerModule, MaxPoolLayer, GlobalAvgPoolLayer,\
    FlattenLayer, DropoutLayer, ConvLayer as _ConvLayer


def acc_at_k(k: int, preds, labels: np.ndarray) -> float:
    top_k_preds = np.stack(preds.apply(lambda row: row.sort_values(ascending=False)[:k].index.tolist(), axis=1))
    acc_at_k = np.mean([labels[i] in top_k_preds[i] for i in range(len(labels))])
    return acc_at_k

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

    print('loading data')
    train_inputs, train_labels, val_inputs, val_labels, test_inputs = load_data('miniplaces')
    n_classes = len(np.unique(train_labels))
    print('data loaded')

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

    data_params = {'na': args.na, 'nb': args.nb, 'nc': args.nc, 'batch_norm': batch_norm, 'drop_prob': args.drop_prob}

    print('making cnn')
    cnn = CNN(layers, n_classes=n_classes, batch_size=128, l2_lambda=args.l2_lambda,
              learning_rate=args.learning_rate, add_scaling=args.scaling, data_params=data_params,
              models_dir=get_abs_path('../miniplaces/models'), config=config)
    print('made cnn; starting training')
    cnn.train(train_inputs, train_labels, val_inputs, val_labels, verbose=1)

    preds = cnn.predict_proba(val_inputs)
    acc1 = acc_at_k(1, preds, val_labels)
    acc5 = acc_at_k(5, preds, val_labels)
    print(f'Args: {args}; acc1: {acc1}; acc5: {acc5}')

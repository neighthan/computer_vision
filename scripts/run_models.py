#! /usr/bin/python

from __future__ import print_function, division, absolute_import
import itertools

import os
import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import VGG
import numpy as np

import argparse

def load_data():
    train_inputs = np.load('../miniplaces/data/images/train.npy')
    train_labels = np.load('../miniplaces/data/images/train_labels.npy')
    val_inputs = np.load('../miniplaces/data/images/val.npy')
    val_labels = np.load('../miniplaces/data/images/val_labels.npy')
    test_inputs = np.load('../miniplaces/data/images/test.npy')
    return train_inputs, train_labels, val_inputs, val_labels, test_inputs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Command line interface for running LSTM models to predict task onset for ICU patients.\
    See usage and argument help for specifics of required/allowed options. An example way to run this script:\
    ./run_models.py -cl 4 -ud phys -t vaso -hn 512 256 -ud phys static topics\
    This runs models with a censor-length of 4 during training, 2 LSTM layers (512 and 256 hidden nodes), predicting vasopressor onset\
    with one model using only physiological data and one using all three data types together. More optional arguments are possible.")
    # parser.add_argument('-n', '--n_attempts', type=int, help="Number of random starts to do of each model; only the results from the best\
    # start will be kept. [default = 3]", default=3)
    parser.add_argument('-bs', '--batch_size', type=int, default=256, help="Batch size for model training. [default = 256]")
    parser.add_argument('-d', '--device', type=int, help="The number of the GPU to use [default = most memory free]")
    parser.add_argument('-hn', '--hidden_nodes', nargs='+', required=True, help="How many hidden nodes to use in the LSTM layer(s).\
    Specifying more than one number causes multiple LSTM layers to be used (e.g. 512, 256 makes a 2 layer LSTM where the 1st layer has\
    512 nodes and the 2nd 256).", type=int, action='append')
    parser.add_argument('-ln', '--log_notes', help="Additional notes to put in the log. You can use this, e.g., to make it\
    easy to identify this set of runs by specifying a note not used elsewhere.", default='')
    parser.add_argument('--test', help="If this flag is given, the arguments are printed and then the script exits. This can be\
    used if you need to make sure your arguments are properly formatted.", action="store_true")
    parser.add_argument('-k', '--key', default='default', help='Key to use in the HDF5 log file when these models are saved.')

    args = parser.parse_args()
    print("Running models with arguments:\n{}\n".format(args))

    if args.test:
        sys.exit()
    
    train_inputs, train_labels, val_inputs, val_labels, test_inputs = load_data()
    n_classes = len(np.unique(train_labels))

    hidden_nodess = args.hidden_nodes
    # n_attempts = args.n_attempts
    batch_size = args.batch_size
    device = args.device
    log_notes = args.log_notes
    log_key = args.key

    # define combinations: one model will be run for each of these
    all_params = itertools.product(*hidden_nodess)

    # next params to use; the ones that, e.g., caused an error before
    # set checkpoint_reached to True if not using this
    checkpoint_reached = True
    checkpoint_params = ()

    while True:
        try:
            params = all_params.next()
            hidden_nodes = params
            print("Using params:", params)
        except StopIteration:
            break

        if not checkpoint_reached:
            if params == checkpoint_params:
                checkpoint_reached = True
            else:
                continue

        # val_aucs = []
        # run_nums = []
        #for attempt_num in xrange(n_attempts):
        #    print("On attempt", attempt_num)

        cnn = VGG(n_classes=n_classes, dense_nodes=hidden_nodes, batch_size=64)
        cnn.train(train_inputs, train_labels, val_inputs, val_labels, verbose=True)

            # val_aucs.append(dev_auc)
            # run_nums.append(run_num)

        # best_idx = np.argmax(val_aucs)
        # best_run_num = run_nums.pop(best_idx) # so we don't delete this one
        # for run_num in run_nums:
        #     os.system('rm -rf {}{}'.format(summaries_path, run_num))

        # TODO: load model, get and save probabilities?

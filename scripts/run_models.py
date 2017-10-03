#! /usr/bin/python

from __future__ import print_function, division, absolute_import
import itertools
from models import PretrainedCNN
import numpy as np
import argparse
import os
import sys
from utils import load_data, tf_init


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Command line interface for running CNN models.\
    See usage and argument help for specifics of required/allowed options. An example way to run this script:")
    # parser.add_argument('-n', '--n_attempts', type=int, help="Number of random starts to do of each model; only the results from the best\
    # start will be kept. [default = 3]", default=3)
    parser.add_argument('-t', '--task', help="Which task to predict on (one of miniplaces or vqa).", required=True)
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help="Batch size for model training. [default = 64]")
    parser.add_argument('-d', '--device', default='', help="The number of the GPU to use [default = most memory free]")
    parser.add_argument('-hn', '--hidden_nodes', nargs='+', required=True, help="How many hidden nodes to use in the dense layer(s).\
    Specifying more than one number causes multiple layers to be used.", type=int, action='append')
    #parser.add_argument('-ln', '--log_notes', help="Additional notes to put in the log. You can use this, e.g., to make it\
    #easy to identify this set of runs by specifying a note not used elsewhere.", default='')
    parser.add_argument('-f', '--finetune', help="Whether to finetune the CNN layers of a network with a pretrained CNN",
                        action='store_true')
    parser.add_argument('-c', '--cnn', help='Which pretrained CNN module to use (vgg16 or xception) [default = vgg16]',
                       default='vgg16')
    parser.add_argument('--test', help="If this flag is given, the arguments are printed and then the script exits. This can be\
    used if you need to make sure your arguments are properly formatted.", action="store_true")
    parser.add_argument('-k', '--key', default='default', help='Key to use in the HDF5 log file when these models are saved.')

    args = parser.parse_args()
    print("Running models with arguments:\n{}\n".format(args))

    if args.test:
        sys.exit()
    
    config = tf_init(args.device)
    train_inputs, train_labels, val_inputs, val_labels, test_inputs = load_data(args.task)
    n_classes = len(np.unique(train_labels))

    # define combinations: one model will be run for each of these
    all_params = itertools.product(args.hidden_nodes)

    # next params to use; the ones that, e.g., caused an error before
    # set checkpoint_reached to True if not using this
    checkpoint_reached = True
    checkpoint_params = ()

    while True:
        try:
            params = all_params.next()
            hidden_nodes = params[0]
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
        #for attempt_num in xrange(args.n_attempts):
        #    print("On attempt", attempt_num)

        cnn = PretrainedCNN(n_classes=n_classes, dense_nodes=hidden_nodes, batch_size=args.batch_size, log_key=args.key,
                           finetune=args.finetune, cnn_module=args.cnn, config=config)
        cnn.train(train_inputs, train_labels, val_inputs, val_labels, verbose=True)

            # val_aucs.append(dev_auc)
            # run_nums.append(run_num)

        # best_idx = np.argmax(val_aucs)
        # best_run_num = run_nums.pop(best_idx) # so we don't delete this one
        # for run_num in run_nums:
        #     os.system('rm -rf {}{}'.format(summaries_path, run_num))

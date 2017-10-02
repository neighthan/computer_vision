from __future__ import division, print_function
import re
import os
import subprocess
import cPickle as pickle
import tensorflow as tf
from PIL import Image
from PIL import PngImagePlugin


def get_best_gpu(n_gpus=4):
    mem_pattern = re.compile("\d+MiB /")
    gpu_usage = {}
    for i in range(n_gpus):
        gpu_i_usage = subprocess.check_output(['nvidia-smi', '--id={}'.format(i)])
        match = mem_pattern.search(gpu_i_usage)
        gpu_usage[i] = int(match.group()[:-5]) # drop "MiB /"

    device = str(sorted(gpu_usage.items(), key=lambda x: x[1])[0][0]) # sort by value (mem usage), take smallest
    return device


def get_abs_path(relative_path):
    """
    :param str relative_path: relative path from this script to a file or directory
    :returns: absolute path to the given file or directory
    :rtype: str
    """
    script_dir = os.path.dirname(__file__)
    return os.path.realpath(os.path.join(script_dir, *relative_path.split(os.path.sep)))


def get_next_run_num(run_num_file, verbose=True):
    """
    Gets the next run number (the one stored in the given file) and increments the number in the file by one
    (run numbers are used to store the saved models and other information for each run)
    :param str run_num_file: the path to the pickled file containing the next available run number
    :param bool verbose: whether to print a message stating the number of the run
    :returns: the run number loaded
    :rtype: int
    """

    try:
        with open(run_num_file) as infile:
            run_num = pickle.load(infile)
    except IOError:  # no file yet
        run_num = 0

    with open(run_num_file, 'w') as outfile:
        pickle.dump(run_num + 1, outfile)

    if verbose:
        print("Beginning run {}.".format(run_num))
    return run_num


def tf_init(device=''):
    """
    Runs common operations at start of TensorFlow:
      - sets logging verbosity to warn
      - sets CUDA visible devices to `device` or, if `device` is '', to the GPU with the most free memory
      - creates a TensorFlow config which allows for GPU memory growth and for soft placement
    :param str device: which GPU to use
    :returns: the aforementioned TensorFlow config
    """

    tf.logging.set_verbosity(tf.logging.WARN)
    if device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_best_gpu()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    return config


def add_metadata_to_img(fname, metadata):
    """
    Adds the given metadata to the png file located at fname.
    This function currently only works for png's.
    :param str fname:
    :param dict metadata:
    :rtype: None
    """
    img = Image.open(fname)
    meta = PngImagePlugin.PngInfo()

    metadata.update(img.info)  # keep previous metadata
    for key in metadata:
        print(key, str(metadata[key]))
        meta.add_text(key, str(metadata[key]))
    img.save(fname, "png", pnginfo=meta)


def read_metadata_from_img(fname):
    """
    :param str fname: path to a png file
    :return: metadata from the png file
    :rtype: dict
    """
    img = Image.open(fname)
    return img.info
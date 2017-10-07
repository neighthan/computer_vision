import re
import os
import subprocess
import pickle
import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import PngImagePlugin


def output_file(preds, split='test', k=5, pad_to=8):
    top_k_preds = np.stack(preds.apply(lambda row: row.sort_values(ascending=False)[:k].index.tolist(), axis=1))
    fnames = ['{}/{}.jpg'.format(split, str(i).zfill(pad_to)) for i in range(1, len(top_k_preds) + 1)]
    preds = pd.DataFrame(top_k_preds, index=fnames)
    preds.to_csv('preds.txt', sep=' ', header=None)


def flatten(list_):
    flat = []
    for elem in list_:
        if type(elem) in [list, tuple]:
            flat.extend(flatten(elem))
        else:
            flat.append(elem)
    return flat


def get_best_gpu(n_gpus=4):
    mem_pattern = re.compile("\d+MiB /")
    gpu_usage = {}
    for i in range(n_gpus):
        gpu_i_usage = subprocess.check_output(['nvidia-smi', '--id={}'.format(i)]).decode('utf8')
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
        with open(run_num_file, 'rb') as infile:
            run_num = pickle.load(infile)
    except IOError:  # no file yet
        run_num = 0

    with open(run_num_file, 'wb') as outfile:
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

    if device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = get_best_gpu()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    return config


def load_data(task):
    if task == 'miniplaces':
        return _load_data_miniplaces()
    elif task == 'vqa':
        return _load_data_vqa()
    assert False, 'task must be one of {{miniplaces, vqa}}, not {}.'.format(task)


def _load_data_miniplaces():
    miniplaces = get_abs_path('../miniplaces/')

    train_inputs = np.load('{}/data/images/train.npy'.format(miniplaces))
    train_labels = np.load('{}/data/images/train_labels.npy'.format(miniplaces))
    val_inputs = np.load('{}/data/images/val.npy'.format(miniplaces))
    val_labels = np.load('{}/data/images/val_labels.npy'.format(miniplaces))
    test_inputs = np.load('{}/data/images/test.npy'.format(miniplaces))
    return train_inputs, train_labels, val_inputs, val_labels, test_inputs


def _load_data_vqa():
    raise NotImplemented


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

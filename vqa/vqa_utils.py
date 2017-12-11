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
from tf_layers.models import NN
from tf_layers.tf_utils import tf_init

from typing import Optional

config = tf_init()

# import sys
# sys.path.append('/afs/csail.mit.edu/u/n/nhunt/github/models/research/slim/')
# from nets import resnet_v2


# DATA LOADING #####

def question_to_ids(question: str, pad_to_length: int, word_to_id):
    unk_val = word_to_id['<UNK>']
    pad_val = word_to_id['<PAD>']
    assert pad_val == 0

    ids = [word_to_id.get(word, unk_val) for word in question.split(' ')]
    return ids + [pad_val] * (pad_to_length - len(ids))


def question_to_idx(question: str, pad_to_length: int, vocab) -> list:
    padding = '<PAD>'

    question = [word if word in vocab else '<unk>' for word in question.split(' ')]
    question += [padding] * (pad_to_length - len(question))
    return question[:pad_to_length]  # crop if needed


def process_objects(objects, max_n_objects: int, object_to_id):
    objects = [[object_to_id[obj[0]], *obj[1:]] for obj in objects]
    return objects + [[0] * 6] * (max_n_objects - len(objects))


def load_data(splits, add_scene_to_objects: bool = True, embed_object_names: bool = True, use_glove: bool = True,
              glove_size: int = 100, return_img_paths: bool = True, max_question_length: int = 22,
              max_n_objects: int = 56, object_limit: int=16,
              n_pose_ids: int = 10, n_scene_types: int = 2) -> list:
    """
    There are some potential issues here with, e.g., max_question_length if this value differs between splits.
    Ideally we should compute the max values on train and then crop anything longer than that. As a workaround,
    though, because I don't do any cropping right now, I've made these into parameters and given them defaults
    that are equal to their max values across all splits.
    """

    #     max_question_length = qa.question.map(lambda question: len(question.split(' '))).max()
    #     max_n_objects = qa.objects.map(len).max()
    #     n_pose_ids = len(np.unique(objects[:, :, 5]))
    #     n_scene_types = len(np.unique(objects[:, :, 6]))

    with open('data/answer_to_id.pkl', 'rb') as f:
        answer_to_id = pickle.load(f)

    if use_glove:
        # csv.QUOTE_NONE = 3; this makes it so things like " are parsed as words instead of as the start of a string
        glove_vecs = pd.read_csv(f'data/glove/glove.6B.{glove_size}d.txt', sep=' ', index_col=0, header=None, quoting=3)
        glove_vecs = pd.concat([glove_vecs, pd.DataFrame([[0] * glove_size], index=['<PAD>'], columns=glove_vecs.columns)])
    else:
        with open('data/word_to_id.pkl', 'rb') as f:
            word_to_id = pickle.load(f)

    with open('data/object_to_id.pkl', 'rb') as f:
        object_to_id = pickle.load(f)

    with open('data/scene_to_id.pkl', 'rb') as f:
        scene_to_id = pickle.load(f)

    # one-hot encode flip, poseID, and possibly object name; keep x, y, z as they are
    tf.reset_default_graph()
    sess = tf.Session(config=config)

    inp = tf.placeholder(tf.int32, (None, max_n_objects, 7))

    flip = tf.one_hot(inp[:, :, 4], 2)
    pose_id = tf.one_hot(inp[:, :, 5], n_pose_ids)
    transforms = [flip, pose_id]

    if add_scene_to_objects:
        scene_type = tf.one_hot(inp[:, :, 6], n_scene_types)
        transforms.append(scene_type)

    if not embed_object_names:
        object_names = tf.one_hot(inp[:, :, 0], max(object_to_id.values()))
        transforms.append(object_names)

    one_hot_inputs = tf.concat((*transforms, tf.cast(inp[:, :, 1:4], tf.float32)), axis=-1)

    all_data = []
    for split in splits:
        qa = pd.read_hdf(f'data/{split}/qa.h5')

        # convert words and answers to ids
        if use_glove:
            question_idx = np.array(
                qa.question.map(lambda question: question_to_idx(question, max_question_length, glove_vecs.index)).tolist())
            questions = glove_vecs.loc[question_idx.reshape(-1)].values.reshape(*question_idx.shape, -1)
        else:
            questions = np.array(
                qa.question.map(lambda question: question_to_ids(question, max_question_length, word_to_id)).tolist())

        if split != 'test':
            unk_val = answer_to_id['<UNK>']
            answers = qa.answer.map(lambda answer: answer_to_id.get(answer, unk_val)).values

        img_paths = qa.img_path.values

        # 0: name 1: x 2: y 3: z 4: flip 5: poseID 6: scene type, if added below
        objects = np.array(
            qa.objects.map(lambda objects: process_objects(objects, max_n_objects, object_to_id)).tolist())

        scenes = qa.scene_type.map(scene_to_id.get).values

        if add_scene_to_objects:
            objects = np.concatenate((objects, np.repeat(scenes, max_n_objects).reshape(-1, max_n_objects, 1)), axis=-1)

        objects_one_hot = sess.run(one_hot_inputs, {inp: objects})

        inputs = {'question': questions, 'objects': objects_one_hot}

        if embed_object_names:
            inputs['object_names'] = objects[:, :, 0]

        if return_img_paths:
            inputs['img_paths'] = img_paths

        inputs = {key: val[:, :object_limit] if 'object' in key else val for key, val in inputs.items()}

        all_data.append(inputs)
        if split != 'test':
            all_data.append({'default': answers})

    sess.close()
    return all_data


def generator(inputs: dict, batch_size: int, split: str, labels: Optional[dict]=None,
              n_imgs_per_file: int=640, n_qa_per_image: int=3, mean=None, std=None, shuffle: bool=True):
    batch_size = int(np.round(batch_size / n_qa_per_image)) # because we'll load all qa for each image at once

    img_dir = f'/cluster/nhunt/img_features/{split}'
    img_files = [f"{img_dir}/{img}" for img in os.listdir(img_dir)]
    if shuffle:
        np.random.shuffle(img_files)

    for img_file in img_files:
        file_number = int(img_file.split('features')[-1].replace('.npy', ''))

        imgs = np.load(img_file)
        if mean is not None:
            imgs -= mean
            imgs /= std

        idx = list(range(len(imgs)))
        if shuffle:
            np.random.shuffle(idx)
        imgs = imgs[idx]

        # offset to get the global indices (for questions and answers)
        idx_offset = file_number * n_imgs_per_file * n_qa_per_image

        n_batches = int(np.ceil(len(idx) / batch_size))
        for batch in range(n_batches):
            batch_idx = idx[batch * batch_size : (batch + 1) * batch_size]
            batch_qa_idx = [idx_offset + img_idx * n_qa_per_image + qa_num for img_idx in batch_idx for qa_num in range(n_qa_per_image)]

            batch_inputs = {key: val[batch_qa_idx] for key, val in inputs.items() if key != 'img_paths'}
            batch_inputs['img'] = imgs[np.repeat(batch_idx, n_qa_per_image)]

            if labels is not None:
                batch_labels = {key: val[batch_qa_idx] for key, val in labels.items()}
                yield batch_inputs, batch_labels
            else:
                yield batch_inputs


def create_predictions_file(model_name: str, models_dir: str, split: str):
    """
    Submission requirements (from here: http://visualqa.org/vqa_v1_challenge.html)
    Before uploading your results to the evaluation server, you will need to create a JSON file containing
    your results in the correct format as described on the evaluation page. The file should be named
    "vqa_[task_type]_[dataset]_[datasubset]_[alg_name]_results.json". Replace [task_type] with either
    "OpenEnded" or "MultipleChoice" depending on the challenge you are participating in, [dataset] with
    either "mscoco" or "abstract_v002" depending on whether you are participating in the challenge for real
    images or abstract scenes, [datasubset] with either "test-dev2015" or "test2015" depending on the test
    split you are using, and [alg] with your algorithm name. Place the JSON file into a zip file named "results.zip".
    """

    inputs = load_data([split])[0]
    object_limit = 16  # > 90% of cases covered in train
    inputs = {key: val[:, :object_limit] if 'object' in key else val for key, val in inputs.items()}

    batch_size = 64
    mean = np.load('data/train/mean.npy')
    std = np.load('data/train/std.npy')
    input_generator = lambda: generator(inputs, batch_size, split, mean=mean, std=std)

    model = NN(model_name=model_name, models_dir=models_dir, config=config)
    predictions = np.concatenate(model._batch(model.predict['default'], generator=input_generator)[0])

    qa = pd.read_hdf(f'data/{split}/qa.h5')

    with open('data/answer_to_id.pkl', 'rb') as f:
        answer_to_id = pickle.load(f)
    id_to_answer = {val: key for key, val in answer_to_id.items()}

    # 1000 is used for choices that we don't have a class for
    choices = np.array(qa.choices.map(lambda choices: [answer_to_id.get(choice, 1000) for choice in choices]).tolist())

    # ensure that the max value for each row is one of those that are allowed
    predictions[np.repeat(np.arange(len(predictions)), choices.shape[1]), choices.ravel()] += 2

    answers = [id_to_answer[pred] for pred in predictions.argmax(axis=1)]

    json_answers = [{'answer': answers[i], 'question_id': int(qa.question_id.iloc[i])} for i in range(len(answers))]

    alg_name = 'test'
    with open(f'vqa_MultipleChoice_abstract_v002_{split}2015_{alg_name}_results.json', 'w') as f:
        json.dump(json_answers, f)


def multiple_choice_accuracy(predictions: np.ndarray, split: str) -> float:
    predictions = predictions.copy() # because we'll be modifying this

    qa = pd.read_hdf(f'data/{split}/qa.h5')

    with open('data/answer_to_id.pkl', 'rb') as f:
        answer_to_id = pickle.load(f)

    # 1000 is used for choices that we don't have a class for
    choices = np.array(qa.choices.map(lambda choices: [answer_to_id.get(choice, 1000) for choice in choices]).tolist())

    # ensure that the max value for each row is one of those that are allowed
    predictions[np.repeat(np.arange(len(predictions)), choices.shape[1]), choices.ravel()] += 2

    # answering 1000 (unk) doesn't actually count for accuracy; use a dummy value that won't ever be predicted
    answers = qa.answer.map(lambda answer: answer_to_id.get(answer, -1))

    return (predictions.argmax(axis=1) == answers).sum() / len(predictions)


# MODEL BUILDING #####

def soft_crop(img, names, xy, n_object_types):
    """
    :param img: placeholder(float32, (batches x img_height x img_width x channels))
    :param names: placeholder(int32, (batches x max_n_objects))
    :param xy: placeholder(float32, (batches x max_n_objects x 2))
    """

    with tf.name_scope('soft_crop'):
        img_height, img_width = img.shape.as_list()[1:3]
        n_samples = tf.shape(img)[0]
        max_n_objects = names.shape.as_list()[1]

        sigmas = tf.nn.embedding_lookup(tf.get_variable('object_sigmas', (n_object_types, 2), initializer=tf.constant_initializer(1)), names)

        gaussians_x = tf.distributions.Normal(xy[:, :, 0], sigmas[:, :, 0], name='gaussian_x')
        gaussians_y = tf.distributions.Normal(xy[:, :, 1], sigmas[:, :, 1], name='gaussian_y')

        x_coords = tf.range(0, img_width, dtype=tf.float32)
        y_coords = tf.range(0, img_height, dtype=tf.float32)

        x_coords = tf.tile(tf.reshape(x_coords, (img_width, 1, 1)), (1, n_samples, max_n_objects), name='x_coords')
        y_coords = tf.tile(tf.reshape(y_coords, (img_height, 1, 1)), (1, n_samples, max_n_objects), name='y_coords')

        x_weights = gaussians_x.prob(x_coords)
        y_weights = gaussians_y.prob(y_coords)

        x_weights /= tf.reduce_max(x_weights, axis=[1, 2], keep_dims=True)
        x_weights = tf.transpose(x_weights, [1, 2, 0])

        y_weights /= tf.reduce_max(y_weights, axis=[1, 2], keep_dims=True)
        y_weights = tf.transpose(y_weights, [1, 2, 0])

        # reshape everything to (batches x objects x height x width x channels)
        x_weights = tf.reshape(x_weights, (n_samples, max_n_objects, 1, img_width, 1))
        y_weights = tf.reshape(y_weights, (n_samples, max_n_objects, img_height, 1, 1))
        img_expanded = tf.expand_dims(img, 1)

        masked_imgs = img_expanded * x_weights * y_weights
        return masked_imgs


def memory_module(question_and_facts, attention_n_units: int=256,
                  attention_activation=tf.nn.tanh, memory_activation=tf.nn.relu, n_glances: int=4):
    """
    facts and question must have the same size final dimension
    :param question: rank 2
    :param facts: rank 3
    """

    question, facts = question_and_facts

    with tf.name_scope('memory_module'):
        # compute things that don't depend on a specific glance here so we don't redo work
        question_exp = tf.expand_dims(question, 1)  # expand so we can broadcast into element-wise operations with each fact
        question_times_facts = question_exp * facts
        question_minus_facts = tf.abs(question_exp - facts)

        last_memory = question  # initialize memory state to be the question; could try other things too

        for i in range(n_glances):
            with tf.name_scope(f'memory_{i}'):
                last_memory_exp = tf.expand_dims(last_memory, 1)

                attention_interactions = tf.concat((
                    question_times_facts,
                    last_memory_exp * facts,
                    question_minus_facts,
                    tf.abs(facts - last_memory_exp)
                ), axis=-1)

                # soft attention: weighted sum of facts
                attention_distribution = tf.layers.dense(attention_interactions, attention_n_units, activation=attention_activation, name='')
                attention_distribution = tf.nn.softmax(tf.layers.dense(attention_distribution, 1, activation=None), name=f'attention_{i}')

                context_vector = tf.reduce_sum(attention_distribution * facts, axis=1, name=f'context_{i}')
                next_memory = tf.layers.dense(tf.concat((last_memory, context_vector, question), axis=-1),
                                              last_memory.shape[-1].value, activation=memory_activation, name=f'memory_{i}')

                last_memory = next_memory

    return next_memory


def bilinear_pool(inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
        output = tf.reduce_sum(tf.matmul(tf.expand_dims(output, -1), tf.expand_dims(inputs[i], -2)), axis=-1)
    return output


def get_layers(n_object_types, max_n_objects, question_embedding_size=150, object_embedding_size=75,
               question_lstm_size=256, dense_size=1024, n_glances=6, include_img=True, input_fusion=False,
               use_bilinear_pool: bool=False):
    # input is questions
    question_module = LayerModule([
        #         EmbeddingLayer(vocab_size, question_embedding_size, name='embed_question'),
        LSTMLayer(question_lstm_size, scope='lstm_question')
    ])

    # input is [object_features, object_names]
    objects_module = LayerModule([
        BranchedLayer([None, EmbeddingLayer(n_object_types, object_embedding_size, name='embed_object')]),
        MergeLayer(axis=-1)
    ])

    # image section; split into global image processing, soft crops for objects, and final processing
    divides_before_soft_crop = 5  # all assumed to be 2x2 reductions

    # img_before_softcrop = LayerModule([
    #     CustomLayer(vgg_preprocessing),
    #     CustomLayer(resnet50)
    # ])

    img_after_softcrop = LayerModule([
        GlobalMaxPoolLayer(),
        FlattenLayer(),
    ])

    # input is [img, object_names, object_features (for xy data)]
    objects_img_module = LayerModule([
        #         BranchedLayer([img_before_softcrop, None, None]),
        CustomLayer(
            lambda inputs: soft_crop(inputs[0], inputs[1], inputs[2][:, :, -3:-1] / 2 ** divides_before_soft_crop, n_object_types)),
        CustomLayer(lambda img: tf.reshape(img, (-1, *img.shape.as_list()[2:]))),
        # put object dim into the batch dimension
        img_after_softcrop,
        CustomLayer(lambda img: tf.reshape(img, (-1, max_n_objects, img.shape[-1].value)))
        # bring object dim back from the batch dimension
    ])

    img_module = LayerModule([
        GlobalAvgPoolLayer(),
        FlattenLayer(),
    ])

    if include_img:
        layers = [
            BranchedLayer([question_module, objects_module, objects_img_module, img_module],
                          layer_input_map={0: [0], 1: [1, 2], 2: [3, 2, 1], 3: [3]}),
            BranchedLayer([None, MergeLayer(axis=-1), None], layer_input_map={0: [0], 1: [1, 2], 2: [3]})
        ]

        if input_fusion:
            layers.append(BranchedLayer(
                [None, LSTMLayer(question_lstm_size, ret='output', last_only=False, scope='input_fusion'), None]))

        layers.extend([
            # TODO: get n_qa from 1st input, tile 2nd input to repeat object vectors for each qa
            BranchedLayer([None, CustomLayer(memory_module, {'n_glances': n_glances}), None],
                          layer_input_map={0: [0], 1: [0, 1], 2: [2]}),
            CustomLayer(bilinear_pool) if use_bilinear_pool else MergeLayer(axis=-1),
            DenseLayer(dense_size, batch_norm=''),
            DropoutLayer(0.7)
        ])
    else:
        layers = [
            BranchedLayer([question_module, objects_module], layer_input_map={0: [0], 1: [1, 2]}),
            #             BranchedLayer([None, DenseLayer(question_lstm_size)]) # make objects fit question size
        ]

        if input_fusion:
            layers.append(BranchedLayer(
                [None, LSTMLayer(question_lstm_size, ret='output', last_only=False, scope='input_fusion')]))

    return layers

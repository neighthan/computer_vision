import tensorflow as tf
from utils import flatten
from typing import Union, Sequence

_int_or_two = Union[int, Sequence[int]]

_activations = {
    'relu': tf.nn.relu
}


class _Layer(object):
    """
    A layer must have the following attributes:
      - self.params: a dictionary that specifies keyword arguments for the layer
      - self.batch_norm: a boolean specifying whether to use batch normalization after this layer
      - self.layer: a tensorflow layer function which accepts self.params as kwargs and input as the first positional argument
    or else the layer must override apply (see, e.g., BranchedLayer).
    """

    def apply(self, inputs):
        output = self.layer(inputs, **self.params)
        if self.batch_norm:
            output = tf.contrib.layers.batch_norm(output)
        return output


class ConvLayer(_Layer):
    def __init__(self, n_filters:int, kernel_size:_int_or_two, strides: int=1,
                 activation: str='relu', padding: str='same', batch_norm: bool=True):
        self.params = dict(
            filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=_activations[activation],
            padding=padding,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
        )
        self.batch_norm = batch_norm
        self.layer = tf.layers.conv2d


class _PoolLayer(_Layer):
    def __init__(self, size:_int_or_two, strides:_int_or_two, padding: str='same'):
        self.params = dict(
            pool_size=size,
            strides=strides,
            padding=padding
        )


class MaxPoolLayer(_PoolLayer):
    def __init__(self, size:_int_or_two, strides: _int_or_two=1, padding: str='same', batch_norm: bool=False):
        super().__init__(size, strides, padding)
        self.batch_norm = batch_norm
        self.layer = tf.layers.max_pooling2d


class AvgPoolLayer(_PoolLayer):
    def __init__(self, size:_int_or_two, strides: _int_or_two=1, padding: str='same', batch_norm: bool=False):
        super().__init__(size, strides, padding)
        self.batch_norm = batch_norm
        self.layer = tf.layers.average_pooling2d


class BranchedLayer(_Layer):
    """
    Takes as input (to .apply) either a single tensor (which will be the input to each layer) or one input per branch.
    If some branches are longer than others, use None as the layer for any non-continuing branches. This will cause the
    input given to be returned as the output as well.
    """

    def __init__(self, layers:Sequence[_Layer]):
        """
        :param List[_Layer] layers:
        """
        self.layers = layers

    def apply(self, inputs):
        if type(inputs) is not list:
            inputs = [inputs] * len(self.layers)
        else:
            assert len(inputs) == len(self.layers)

        outputs = []
        for i in range(len(inputs)):
            if self.layers[i] is not None:
                outputs.append(self.layers[i].apply(inputs[i]))
            else:
                outputs.append(inputs[i])
        return outputs


class MergeLayer(_Layer):
    """
    Takes a BranchedLayer and merges it back into one.
    """

    def __init__(self, axis:int):
        self.params = dict(axis=axis)
        self.layer = tf.concat

    def apply(self, inputs):
        return self.layer(flatten(inputs), **self.params)


class FlattenLayer(_Layer):
    def __init__(self):
        self.params = {}
        self.batch_norm = False
        self.layer = tf.contrib.layers.flatten


class DenseLayer(_Layer):
    def __init__(self, units:int, activation: str='relu', batch_norm: bool=True):
        self.params = dict(
            units=units,
            activation=_activations[activation],
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
        )
        self.batch_norm = batch_norm
        self.layer = tf.layers.dense


class LayerModule(_Layer):
    """
    A set of layers that can be applied as a group (useful if you want to use them in multiple places).
    """

    def __init__(self, layers: Sequence[_Layer]):
        self.layers = layers

    def apply(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.apply(output)
        return output

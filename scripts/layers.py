import tensorflow as tf
from utils import flatten
from typing import Union, Sequence, Optional

_OneOrMore = lambda type_: Union[type_, Sequence[type_]]

_activations = {
    'relu': tf.nn.relu
}

_layers = {
    'conv2d': tf.layers.conv2d,
    'max_pooling2d': tf.layers.max_pooling2d,
    'average_pooling2d': tf.layers.average_pooling2d,
    'flatten': tf.contrib.layers.flatten,
    'dense': tf.layers.dense,
    'dropout': tf.layers.dropout,
    'concat': tf.concat
}

_initializers = {
    'variance_scaling_initializer': tf.contrib.layers.variance_scaling_initializer
}


class _Layer(object):
    """
    A layer must have the following attributes:
      - self.params: a dictionary that specifies keyword arguments for the layer
      - self.batch_norm: a boolean specifying whether to use batch normalization after this layer
      - self.layer: a tensorflow layer function which accepts self.params as kwargs and input as the first positional argument
    or else the layer must override apply (see, e.g., BranchedLayer).
    """

    def apply(self, inputs: tf.Tensor, is_training: tf.Tensor) -> tf.Tensor:
        """

        :param inputs:
        :param is_training:
        :return:
        """
        params = self.params.copy()
        if 'kernel_initializer' in params.keys():
            params['kernel_initializer'] = _initializers[params['kernel_initializer']]()

        output = self.layer(inputs, **params)

        if self.batch_norm:
            output = tf.layers.batch_normalization(output, training=is_training)
        return output


class ConvLayer(_Layer):
    def __init__(self, n_filters:int, kernel_size:_OneOrMore(int), strides: int=1,
                 activation: str='relu', padding: str='same', batch_norm: bool=True):
        self.params = dict(
            filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=_activations[activation],
            padding=padding,
            kernel_initializer='variance_scaling_initializer'
        )
        self.batch_norm = batch_norm

    @property
    def layer(self):
        return _layers['conv2d']


class _PoolLayer(_Layer):
    def __init__(self, size:_OneOrMore(int), strides: _OneOrMore(int)=1, padding: str='same', batch_norm: bool=False):
        self.params = dict(
            pool_size=size,
            strides=strides,
            padding=padding
        )
        self.batch_norm = batch_norm


class MaxPoolLayer(_PoolLayer):
    @property
    def layer(self):
        return _layers['max_pooling2d']


class AvgPoolLayer(_PoolLayer):
    @property
    def layer(self):
        return _layers['average_pooling2d']


class _GlobalPoolLayer(_Layer):
    def __init__(self, batch_norm: bool=False):
        self.params = dict(
            strides=1,
            padding='valid'
        )
        self.batch_norm = batch_norm

    def apply(self, inputs: tf.Tensor, is_training: tf.Tensor):
        params = self.params.copy()
        params['pool_size'] = inputs.shape.as_list()[1:3] # height and width

        output = self.layer(inputs, **params)

        if self.batch_norm:
            output = tf.layers.batch_normalization(output, training=is_training)
        return output


class GlobalAvgPoolLayer(_GlobalPoolLayer):
    @property
    def layer(self):
        return _layers['average_pooling2d']


class GlobalMaxPoolLayer(_GlobalPoolLayer):
    @property
    def layer(self):
        return _layers['max_pooling2d']


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

    def apply(self, inputs: _OneOrMore(tf.Tensor), is_training: tf.Tensor) -> Sequence[tf.Tensor]:
        """

        :param inputs:
        :param is_training:
        :return:
        """
        if type(inputs) is not list:
            inputs = [inputs] * len(self.layers)
        else:
            assert len(inputs) == len(self.layers)

        outputs = []
        for i in range(len(inputs)):
            if self.layers[i] is not None:
                outputs.append(self.layers[i].apply(inputs[i], is_training))
            else:
                outputs.append(inputs[i])
        return outputs


class MergeLayer(_Layer):
    """
    Takes a BranchedLayer and merges it back into one.
    """

    def __init__(self, axis:int):
        self.params = dict(axis=axis)

    @property
    def layer(self):
        return _layers['concat']

    def apply(self, inputs: Sequence[tf.Tensor], is_training: Optional[tf.Tensor]=None) -> tf.Tensor:
        """

        :param inputs: may be arbitrarily nested
        :param is_training: unused
        :returns:
        """
        return self.layer(flatten(inputs), **self.params)


class FlattenLayer(_Layer):
    def __init__(self):
        self.params = {}
        self.batch_norm = False

    @property
    def layer(self):
        return _layers['flatten']


class DenseLayer(_Layer):
    def __init__(self, units:int, activation: str='relu', batch_norm: bool=True):
        self.params = dict(
            units=units,
            activation=_activations[activation],
            kernel_initializer='variance_scaling_initializer'
        )
        self.batch_norm = batch_norm

    @property
    def layer(self):
        return _layers['dense']


class DropoutLayer(_Layer):
    def __init__(self, rate:float):
        self.params = dict(rate=rate)

    @property
    def layer(self):
        return _layers['dropout']

    def apply(self, inputs: tf.Tensor, is_training:tf.Tensor) -> tf.Tensor:
        params = self.params.copy()
        params['training'] = is_training

        return self.layer(inputs, **params)


class LayerModule(_Layer):
    """
    A set of layers that can be applied as a group (useful if you want to use them in multiple places).
    """

    def __init__(self, layers: Sequence[_Layer]):
        self.layers = layers

    def apply(self, inputs: _OneOrMore(tf.Tensor), is_training: tf.Tensor) -> tf.Tensor:
        """

        :param inputs: should be a single tensor unless the first layer in the module is a branch layer; then it can
                       be one tensor per branch (or still a single tensor which is the input to each branch)
        :param is_training:
        :returns:
        """

        output = inputs
        for layer in self.layers:
            output = layer.apply(output, is_training)
        return output

# https://toolz.readthedocs.io/en/latest/api.html#dicttoolz might be fun

from attrdict import AttrDict
from functools import reduce
from operator import mul
import tensorflow as tf
import numpy as np

K = tf.keras
L = K.layers

# sensors:
# 2D = (H, W, C) ---> (? ? ?)
# 1D = (W, C) ---> (? ? ?)
# str = string ---> (? ? ?)

# actuators:
# onehot = (? ? ?) ---> (N, ) one 1 rest 0
# discrete = int from 0:N
# box usually float (W, C) maybe has low and high
# ragged output requires SHAPE VARIABLES


def get_shape(tensor, batch=False):
    tensor = tf.squeeze(tensor, 0) if batch else tensor
    if tf.executing_eagerly():
        return tensor.shape
    else:
        return tf.shape(tensor)
    raise Exception("cannot get shape for:", tensor)


def get_size(shape):
    if isinstance(shape, tf.tensor):
        return tf.math.reduce_prod(shape)
    elif isinstance(shape, tuple):
        return reduce(mul, shape)
    else:
        raise Exception("cannot get size for shape:", shape)


def get_format(item, format=None):
    if isinstance(format, str):
        return format
    if isinstance(item, tf.tensor):
        return f"tf {str(item.numpy().dtype)}"
    elif isinstance(item, np.ndarray):
        return f"np {str(item.dtype)}"


def get_spec(tensor, format=None, batch=False):
    """ make an attrdict with an tensor's shape, size, and format """
    shape = get_shape(tensor, batch)
    return AttrDict({
        "shape": shape,
        "size": get_size(shape),
        "format": get_format(tensor, format)})


def get_conv2d(f, hp):
    return L.Conv2D(hp.conv2d_filters, hp.conv2d_kernel_size,
                    activation=hp.conv2d_activation)


def flatten_resize_reshape(output, out_shape, hp):
    output = tf.concat([tf.range(output.shape[0]), output], -1)
    output = L.Flatten()(output)
    output = L.Dense(get_size(out_shape))(output)
    output = L.Reshape(out_shape)(output)
    assert output.shape == out_shape
    return output


@tf.function
def get_cartesian_product(a, b, normalize=True):
    a = tf.range(a)
    a = tf.cast(a, tf.float32)
    b = tf.range(b)
    if normalize:
        a = a / tf.math.reduce_max(a)
        b = b / tf.math.reduce_max(b)
    b = tf.cast(b, tf.float32)
    return tf.reshape(tf.stack(tf.meshgrid(a, b, indexing='ij'), axis=-1), (-1, 2))


@tf.function
def concat_coords(tensor):
    """ (H, W, C) ---> (H, W, C+2) with i,j coordinates"""
    in_shape = tf.shape(tensor)
    coords = get_cartesian_product(*in_shape)
    return tf.concat([tensor, coords], -1)


def two_to_one(output, out_shape, hp):
    """ (H, W, C) ---> (W, C)"""
    output = concat_coords(output)
    for i in range(hp.two_to_one_layers):
        output = get_conv2d(hp)(output)
    output = flatten_resize_reshape(output, out_shape)
    assert output.shape == out_shape
    return output


def get_interface(in_spec, out_spec, hp):
    """ return a keras model to convert in_spec into out_spec """
    in_shape, out_shape = in_spec.shape, out_spec.shape
    input = K.Input(in_shape)
    if len(in_shape) is 3 and len(out_shape) is 2:
        interface_function = two_to_one
    elif len(in_shape) is 2 and len(out_shape) is 2:
        interface_function = flatten_resize_reshape
    else:
        raise Exception(f"no interface for {str(in_spec)}--->{str(out_spec)}")
    output = interface_function(input, out_spec, hp)
    assert output.shape == out_shape
    output = K.activations.softmax(output) if out_spec.format is "onehot" else output
    return K.Model(input, output)

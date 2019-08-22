import tensorflow as tf

from nature import use_dense
from tools import get_size

K = tf.keras
L = K.layers


def use_flatten_resize_reshape(out_shape):
    flatten = L.Flatten()
    dense = use_dense(get_size(out_shape))
    reshape = L.Reshape(out_shape)

    def call(x):
        return reshape(dense(flatten(x)))
    return call

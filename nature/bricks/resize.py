import tensorflow as tf

from nature import use_linear
from tools import get_size

K = tf.keras
L = K.layers


def use_resizer(out_shape):
    flatten = L.Flatten()
    resize = use_linear(get_size(out_shape))
    reshape = L.Reshape(out_shape)

    def call(x):
        x = flatten(x)
        x = resize(x)
        x = reshape(x)
        return x
    return call


# def use_resizer(out_shape):
#     return K.Sequential([
#         L.Flatten(),
#         use_linear(get_size(out_shape)),
#         L.Reshape(out_shape)
#     ])

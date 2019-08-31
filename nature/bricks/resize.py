import tensorflow as tf

from nature import Linear, Brick
from tools import get_size

K = tf.keras
L = K.layers


def Resizer(agent, out_shape, norm_preact=True):
    flatten = L.Flatten()
    resize = Linear(agent, get_size(out_shape))
    reshape = L.Reshape(out_shape)

    def call(self, x):
        x = flatten(x)
        x = resize(x)
        return reshape(x)
    return Brick(flatten, resize, reshape, call, agent)


# def use_resizer(out_shape):
#     return K.Sequential([
#         L.Flatten(),
#         use_linear(get_size(out_shape)),
#         L.Reshape(out_shape)
#     ])

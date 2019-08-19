import tensorflow as tf

from nurture.bricks.helpers.regularize import get_l1_l2
from nurture.bricks.helpers.chaos import get_chaos
# from .noisedrop import NoiseDrop

from tools.get_brick import get_brick

K = tf.keras
B, L = K.backend, K.layers

LAYER_CLASS = L.Dense  # NoiseDrop
UNITS = 256


def get_dense_out(
        agent, id, input,
        layer_class=LAYER_CLASS, units=UNITS, return_brick=False):
    layer = layer_class(
            units,
            kernel_regularizer=get_l1_l2(),
            activity_regularizer=get_l1_l2(),
            bias_regularizer=get_l1_l2(),
            kernel_initializer=get_chaos(),
            bias_initializer=get_chaos(bias=True),
        )

    def use_layer(x):
        return layer(x)
    return get_brick(use_layer, input, return_brick)

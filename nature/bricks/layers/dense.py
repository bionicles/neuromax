import tensorflow as tf

from nature import get_l1_l2, get_chaos

K = tf.keras
L = K.layers

LAYER_CLASS = L.Dense  # NoiseDrop
UNITS = 128


def use_dense(layer_class=LAYER_CLASS, units=UNITS):
    return layer_class(
            units,
            kernel_regularizer=get_l1_l2(),
            activity_regularizer=get_l1_l2(),
            bias_regularizer=get_l1_l2(),
            kernel_initializer=get_chaos(),
            bias_initializer=get_chaos(bias=True),
        )

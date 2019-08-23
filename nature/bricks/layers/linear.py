import tensorflow as tf

from nature import get_l1_l2, get_chaos

K = tf.keras
L = K.layers

INITIALIZER = get_chaos
REGULARIZER = get_l1_l2
LAYER_CLASS = L.Dense  # NoiseDrop
UNITS = 128


def use_linear(units=UNITS, regularizer=REGULARIZER, initializer=INITIALIZER):
    """
    Get a dense layer with no activation (use_fn does that)
    kwargs:
        units: int for the number of units
        regularizer: callable to provide the regularizer
        initializer: callable to provide the kernel and bias
    """
    return L.Dense(
            units=units,
            kernel_regularizer=regularizer(),
            activity_regularizer=regularizer(),
            bias_regularizer=regularizer(),
            kernel_initializer=initializer(),
            bias_initializer=initializer(bias=True))

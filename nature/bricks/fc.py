import tensorflow as tf

from nature import L1L2, Init

K = tf.keras
L = K.layers

INITIALIZER = Init
REGULARIZER = L1L2
LAYER_CLASS = L.Dense  # NoiseDrop
UNITS = 2048


def FC(units=UNITS, regularizer=REGULARIZER, initializer=INITIALIZER):
    """
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
            bias_initializer=initializer())

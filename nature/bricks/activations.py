import tensorflow as tf

from tools.log import log

K = tf.keras.backend


def gaussian(x):
    return K.exp(-K.pow(x, 2))


def swish(x):
    """
    Searching for Activation Functions
    https://arxiv.org/abs/1710.05941
    """
    return (K.sigmoid(x) * x)


def lisht(x):
    """
    LiSHT: Non-Parametric Linearly Scaled Hyperbolic Tangent
    https://github.com/swalpa/LiSHT
    """
    return (K.tanh(x) * x)


def clean_activation(activation):
    log("clean_activation", activation)
    if activation == 'gaussian':
        return gaussian
    elif activation == 'swish':
        return swish
    elif activation == 'lisht':
        return lisht
    elif activation == 'sin':
        return tf.math.sin
    elif activation == 'cos':
        return tf.math.cos
    else:
        return activation

import tensorflow as tf

K = tf.keras.backend

RRELU_MIN, RRELU_MAX = 0.123, 0.314
HARD_MIN, HARD_MAX = -1., 1.


def clean_activation(activation):
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


def swish(x):
    """
    Searching for Activation Functions
    https://arxiv.org/abs/1710.05941
    """
    return (K.sigmoid(x) * x)


def gaussian(x):
    return K.exp(-K.pow(x, 2))


def hard_tanh(x, min=HARD_MIN, max=HARD_MAX):
    if x > max:
        return max
    elif x < min:
        return min
    else:
        return x


def lisht(x):
    """
    LiSHT: Non-Parametric Linearly Scaled Hyperbolic Tangent
    https://github.com/swalpa/LiSHT
    """
    return (K.tanh(x) * x)


def rrelu(x, min=RRELU_MIN, max=RRELU_MAX):
    return x if x >= 0 else tf.random.uniform(min, max) * x


def tanhshrink(x):
    return x - K.tanh(x)


def hardshrink(x, min=HARD_MIN, max=HARD_MAX):
    if x > max:
        return x
    elif x < min:
        return min
    else:
        return 0

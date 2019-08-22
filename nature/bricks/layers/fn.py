import tensorflow as tf

K = tf.keras

B, L = K.backend, K.layers

RRELU_MIN, RRELU_MAX = 0.123, 0.314
HARD_MIN, HARD_MAX = -1., 1.
SOFT_ARGMAX_BETA = 1e10
FN = 'swish'


def use_fn(fn):
    return L.Activation(clean_activation(fn))


def clean_activation(activation):
    if callable(activation):
        return activation
    elif activation == 'soft_argmax':
        fn = soft_argmax
    elif activation == 'gaussian':
        fn = gaussian
    elif activation == 'swish':
        fn = swish
    elif activation == 'lisht':
        fn = lisht
    elif activation == 'sin':
        fn = tf.math.sin
    elif activation == 'cos':
        fn = tf.math.cos
    else:
        fn = activation
    return fn


def swish(x):
    """
    Searching for Activation Functions
    https://arxiv.org/abs/1710.05941
    """
    return (B.sigmoid(x) * x)


def soft_argmax(x, beta=SOFT_ARGMAX_BETA):
    """
    https://stackoverflow.com/questions/46926809/getting-around-tf-argmax-which-is-not-differentiable
    https://lucehe.github.io/differentiable-argmax/
    """
    x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
    return tf.math.reduce_sum(
        tf.nn.softmax(x * beta) * x_range, axis=-1)


def gaussian(x):
    return B.exp(-B.pow(x, 2))


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
    return (B.tanh(x) * x)


def rrelu(x, min=RRELU_MIN, max=RRELU_MAX):
    return x if x >= 0 else tf.random.uniform(min, max) * x


def tanhshrink(x):
    return x - B.tanh(x)


def hardshrink(x, min=HARD_MIN, max=HARD_MAX):
    if x > max:
        return x
    elif x < min:
        return min
    else:
        return 0

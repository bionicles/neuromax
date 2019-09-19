from tensorflow_addons.activations import sparsemax
import tensorflow as tf
import random

from nature import Polynomial, Logistic, Linear, PSwish, PolySwish
from tools import make_id

K = tf.keras
B, L = K.backend, K.layers

RRELU_MIN, RRELU_MAX = 0.123, 0.314
HARD_MIN, HARD_MAX = -1., 1.
SOFT_ARGMAX_BETA = 1e10
DEFAULT = 'tanh'

@tf.function
def swish(x):
    """
    Searching for Activation Functions
    https://arxiv.org/abs/1710.05941
    """
    return (x * B.sigmoid(x))


@tf.function
def hard_swish(x):
    """
    Searching for MobileNetV3
    https://arxiv.org/abs/1905.02244
    """
    return (x * B.hard_sigmoid(x))

@tf.function
def mish(x):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    https://arxiv.org/abs/1908.08681v1
    """
    return (x * B.tanh(B.softplus(x)))


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

@tf.function
def hard_tanh(x, min=HARD_MIN, max=HARD_MAX):
    if x > max:
        return max * tf.ones_like(x)
    elif x < min:
        return min * tf.ones_like(x)
    else:
        return x

@tf.function
def hard_lisht(x, min=HARD_MIN, max=HARD_MAX):
    if x < min or x > max:
        return max
    else:
        return tf.math.abs(x)


def lisht(x):
    """
    LiSHT: Non-Parametric Linearly Scaled Hyperbolic Tangent
    https://github.com/swalpa/LiSHT
    """
    return (B.tanh(x) * x)


def rrelu(x, min=RRELU_MIN, max=RRELU_MAX):
    return x if x >= 0 else tf.random.uniform(min, max) * x


def tanh_shrink(x):
    return x - B.tanh(x)


def hard_shrink(x, min=HARD_MIN, max=HARD_MAX):
    if x > max:
        return x
    elif x < min:
        return min
    else:
        return 0


FN_LOOKUP = {
    'identity': tf.identity,
    'inverse': lambda x: -x,
    'soft_argmax': soft_argmax,
    'log_softmax': tf.nn.log_softmax,
    'softmax': 'softmax',
    'softplus': B.softplus,
    'sigmoid': B.sigmoid,
    'hard_sigmoid': B.hard_sigmoid,
    'softsign': B.softsign,
    'sparsemax': sparsemax,
    'hard_lisht': hard_lisht,
    'hard_shrink': hard_shrink,
    'tanh_shrink': tanh_shrink,
    'hard_lisht': hard_lisht,
    'hard_tanh': hard_tanh,
    'gaussian': gaussian,
    'swish': swish,
    'hard_swish': hard_swish,
    'mish': mish,
    'lisht': lisht,
    'rrelu': rrelu,
    'relu': B.relu,
    'lrelu': tf.nn.leaky_relu,
    'crelu': tf.nn.crelu,
    'relu6': tf.nn.relu6,
    'selu': tf.nn.selu,
    'elu': B.elu,
    'sin': tf.math.sin,
    'sinh': tf.math.sinh,
    'asinh': tf.math.asinh,
    'cos': tf.math.cos,
    'acos': tf.math.acos,
    'acosh': tf.math.acosh,
    'tan': tf.math.tan,
    'tanh': B.tanh,
    'atan': tf.math.atan,
    'atanh': tf.math.atanh,
    'abs': B.abs,
    'exp': B.exp,
    'expm1': tf.math.expm1,
    'square': B.square,
    'sign': B.sign,
    'sqrt': B.sqrt,
    'log': B.log,
    'digamma': tf.math.digamma,
    'lgamma': tf.math.lgamma,
    'reciprocal': tf.math.reciprocal,
    'rsqrt': tf.math.rsqrt,
}

LAYERS = {
    'polynomial': Polynomial,
    'polyswish': PolySwish,
    'pswish': PSwish,
    'logistic': Logistic,
    'prelu': L.PReLU,
    'linear': Linear
}


def random_fn():
    return FN_LOOKUP[random.choice(FN_LOOKUP.keys())],


def clean_activation(key):
    if callable(key):
        return key
    return FN_LOOKUP[key]


def Fn(key=DEFAULT):
    if key is None:
        key = 'identity'
    if key in LAYERS.keys():
        return LAYERS[key]()
    return L.Activation(clean_activation(key.lower()), name=make_id(key))

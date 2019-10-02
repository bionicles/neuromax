from tensorflow_addons.activations import sparsemax, gelu
import tensorflow as tf
import random

from nature import Polynomial, Logistic, Linear, PSwish, PolySwish, LogisticMap
from tools import make_id

K = tf.keras
B, L = K.backend, K.layers

RRELU_MIN, RRELU_MAX = 0.123, 0.314
HARD_MIN, HARD_MAX = -1., 1.
SOFT_ARGMAX_BETA = 1e10
DEFAULT = 'tanh'


@tf.function(experimental_relax_shapes=True)
def logistic_map(x):
    r = tf.random.truncated_normal((), mean=3.57, stddev=0.005)
    min = tf.math.reduce_min(x)
    x = (x - min) / (tf.math.reduce_max(x) - min)
    return r * x * (1. - x)


@tf.function(experimental_relax_shapes=True)
def swish(x):
    """
    Searching for Activation Functions
    https://arxiv.org/abs/1710.05941
    """
    return (x * B.sigmoid(x))


@tf.function(experimental_relax_shapes=True)
def hard_swish(x):
    """
    Searching for MobileNetV3
    https://arxiv.org/abs/1905.02244
    """
    return (x * B.hard_sigmoid(x))


@tf.function(experimental_relax_shapes=True)
def mish(x):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    https://arxiv.org/abs/1908.08681v1
    """
    return (x * B.tanh(B.softplus(x)))


@tf.function(experimental_relax_shapes=True)
def soft_argmax(x, beta=SOFT_ARGMAX_BETA):
    """
    https://stackoverflow.com/questions/46926809/getting-around-tf-argmax-which-is-not-differentiable
    https://lucehe.github.io/differentiable-argmax/
    """
    x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
    return tf.math.reduce_sum(
        tf.nn.softmax(x * beta) * x_range, axis=-1)


@tf.function(experimental_relax_shapes=True)
def gaussian(x):
    return B.exp(-B.pow(x, 2))


@tf.function(experimental_relax_shapes=True)
def hard_tanh(x, min=HARD_MIN, max=HARD_MAX):
    if x > max:
        return max * tf.ones_like(x)
    elif x < min:
        return min * tf.ones_like(x)
    else:
        return x


@tf.function(experimental_relax_shapes=True)
def hard_lisht(x, min=HARD_MIN, max=HARD_MAX):
    if x < min or x > max:
        return max
    else:
        return tf.math.abs(x)


@tf.function(experimental_relax_shapes=True)
def lisht(x):
    """
    LiSHT: Non-Parametric Linearly Scaled Hyperbolic Tangent
    https://github.com/swalpa/LiSHT
    """
    return (B.tanh(x) * x)


# @tf.function(experimental_relax_shapes=True)
# def gelu(x):
#     return 0.5 * x * (1 + B.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))


@tf.function(experimental_relax_shapes=True)
def fast_gelu(x):
    return B.hard_sigmoid(1.702 * x) * x


@tf.function(experimental_relax_shapes=True)
def rrelu(x, min=RRELU_MIN, max=RRELU_MAX):
    return x if x >= 0 else tf.random.uniform(min, max) * x


@tf.function(experimental_relax_shapes=True)
def tanhshrink(x):
    return x - B.tanh(x)


@tf.function(experimental_relax_shapes=True)
def hardshrink(x, min=HARD_MIN, max=HARD_MAX):
    if x > max:
        return x
    elif x < min:
        return min
    else:
        return 0


# https://pdfs.semanticscholar.org/4be0/701f2d2a34ffb746595bcb251c091ff4a702.pdf
# best on cifar 100:
@tf.function(experimental_relax_shapes=True)
def evolved_1(x):
    if x < 0.:
        x = B.hard_sigmoid(x) * B.elu(x)
    elif x >= 0.:
        x = B.hard_sigmoid(x) * tf.nn.selu(x)
    return x


# best on cifar_10:
@tf.function(experimental_relax_shapes=True)
def evolved_2(x):
    if x < 0.:
        x = B.softplus(x) * B.elu(x)
    elif x >= 0.:
        x = B.hard_sigmoid(x) * x
    return x


FNS = {
    'identity': tf.identity,
    'evolved_1': evolved_1,
    'evolved_2': evolved_2,
    'inverse': lambda x: -x,
    'soft_argmax': soft_argmax,
    'random_logistic_map': logistic_map,
    'log_softmax': tf.nn.log_softmax,
    'softmax': 'softmax',
    'softplus': B.softplus,
    'sigmoid': B.sigmoid,
    'hard_sigmoid': B.hard_sigmoid,
    'softsign': B.softsign,
    'sparsemax': sparsemax,
    'hard_lisht': hard_lisht,
    'hard_shrink': hardshrink,
    'tanh_shrink': tanhshrink,
    'hard_lisht': hard_lisht,
    'hard_tanh': hard_tanh,
    'gaussian': gaussian,
    'swish': swish,
    'hard_swish': hard_swish,
    'mish': mish,
    'lisht': lisht,
    'rrelu': rrelu,
    'fast_gelu': fast_gelu,
    'gelu': gelu,
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
    'logistic_map': LogisticMap,
    'polynomial': Polynomial,
    'polyswish': PolySwish,
    'pswish': PSwish,
    'logistic': Logistic,
    'prelu': L.PReLU,
    'linear': Linear,
}


def random_fn():
    return FNS[random.choice(FNS.keys())],


def clean_activation(key):
    if callable(key):
        return key
    return FNS[key]


def Fn(AI, key=DEFAULT):
    if key is None:
        key = DEFAULT
    #     key = AI.pull("fn", list(LAYERS.keys()) + list(FNS.keys()))
    if key in LAYERS.keys():
        return LAYERS[key]()
    return L.Activation(clean_activation(key.lower()), name=make_id(key))

from tensorflow_addons.activations import sparsemax
import tensorflow as tf

K = tf.keras

B, L = K.backend, K.layers

RRELU_MIN, RRELU_MAX = 0.123, 0.314
HARD_MIN, HARD_MAX = -1., 1.
SOFT_ARGMAX_BETA = 1e10
FN = 'lrelu'


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


FUNCTION_LOOKUP = {
    'soft_argmax': soft_argmax,
    'log_softmax': tf.nn.log_softmax,
    'sparsemax': sparsemax,
    'hard_lisht': hard_lisht,
    'hard_shrink': hard_shrink,
    'tanh_shrink': tanh_shrink,
    'hard_lisht': hard_lisht,
    'hard_tanh': hard_tanh,
    'gaussian': gaussian,
    'swish': swish,
    'lisht': lisht,
    'rrelu': rrelu,
    'lrelu': tf.nn.leaky_relu,
    'crelu': tf.nn.crelu,
    'relu6': tf.nn.relu6,
    'sin': tf.math.sin,
    'cos': tf.math.cos,
}


def clean_activation(activation):
    if callable(activation):
        return activation
    else:
        fn = activation
    return fn


def use_fn(fn):
    if not fn:
        fn = FN
    fn = clean_activation(fn)
    return L.Activation(fn)

import tensorflow as tf

from .norm_preact import get_norm_preact_out
from .layers.dense import get_dense_out

from tools.get_brick import get_brick

K = tf.keras
L = K.layers

SKIP_CLASS = L.Add


def OP_FN(agent, id, out):
    return get_norm_preact_out(agent, id, out, layer_fn=get_dense_out)


def get_skip_out(agent, id, input, op_fn=OP_FN, skip_class=SKIP_CLASS,
                 return_brick=False):
    _, op = op_fn(agent, id, input, return_brick=True)
    skip = skip_class()

    def get_skip(input):
        out = op(agent, id, input)
        return skip([input, out])
    return get_brick(get_skip, input, return_brick)

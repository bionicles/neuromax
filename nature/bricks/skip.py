import tensorflow as tf

from nature import use_norm_preact, use_dense
from tools import make_uuid

K = tf.keras
L = K.layers

SKIP_CLASS = L.Add


def OP_FN(agent, id, out):
    return use_norm_preact(agent, id, out, layer_fn=use_dense)


def use_skip(agent, id, input, op_fn=OP_FN, skip_class=SKIP_CLASS,
             return_brick=False):
    id = make_uuid([id, "skip"])
    _, op = op_fn(agent, id, input, return_brick=True)
    skipper = skip_class()
    parts = dict(op=op, skipper=skipper)

    def call(x):
        y = op(x)
        return skipper([x, y])
    return agent.build_brick(id, parts, call, input, return_brick)

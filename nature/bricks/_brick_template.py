import tensorflow as tf

from nature import use_dense
from tools import make_uuid

K = tf.keras
L = K.layers

LAYER_CLASS = L.Dense
KWARG = 1


def get_xxx_out(agent, id, input, layer_class=LAYER_CLASS, return_brick=False):
    """xxx returns yyy because zzz"""
    id = make_uuid([id, "xxx"])

    part1 = layer_class()
    _, part2 = use_dense(agent, id, input, return_brick=True)
    parts = dict(part1=part1, part2=part2)

    def call(x):
        x = part1(x)
        x = part2(x)
        return x
    return agent.pull_brick(parts)

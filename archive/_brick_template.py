import tensorflow as tf

from nature import use_linear
from tools import log

K = tf.keras
L = K.layers

LAYER_CLASS = L.Dense
KWARG = 1


# TLDR: bricks are auto-generated layers.
# just make layers then use them
# "parts" attrdict is unpacked to make a Brick which extends tf.keras Layer
def use_xxx(agent, parts):
    """xxx does yyy because zzz"""

    log("make xxx layers")
    parts.part1 = part1 = parts.layer_fn()
    parts.part2 = part2 = use_linear(parts.units)

    def call(self, x):
        x = part1(x)
        x = part2(x)
        return x
    parts.call = call
    return parts

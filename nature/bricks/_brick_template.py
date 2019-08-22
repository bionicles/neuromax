import tensorflow as tf

from nature import use_dense

K = tf.keras
L = K.layers

LAYER_CLASS = L.Dense
KWARG = 1


def use_xxx(parts):
    """xxx does yyy because zzz"""

    parts.part1 = part1 = parts.layer_fn()
    parts.part2 = part2 = use_dense(parts.units)

    def call(x):
        x = part1(x)
        x = part2(x)
        return x
    parts.call = call
    return parts

import tensorflow as tf

from tools.get_brick import get_brick

K = tf.keras
L = K.layers

BRICK_BUILDER = L.Dense
KWARG = 1


def get_xxx_out(
        agent, id, input, brick_builder=BRICK_BUILDER, return_brick=False):
    brick = brick_builder()

    def use_brick(x):
        return brick(x)

    return get_brick(use_brick, input, return_brick)

# ragged_actuator.py - bion
# decode sequences as simply as possible

import tensorflow as tf

from nature import use_mlp

K = tf.keras
B, L = K.backend, K.layers

CONCAT_AXIS = -1
FN = "tanh"


def use_ragged_actuator(agent, spec):
    d_in = agent.code_spec.size
    d_out = spec.shape[-1]

    units = d_in ** 2 + 1  # add one to account for the atom
    layer_list = [(units, FN), (units, FN), (d_out, FN)]
    mlp = use_mlp(layer_list=layer_list)

    def call(code, coords):
        return tf.map_fn(
                lambda coord: mlp(
                        tf.concat([code, coord], CONCAT_AXIS)
                    ),
                coords)
    return call

# ragged_actuator.py - bion
# decode sequences as simply as possible

import tensorflow as tf

from tools import make_id

K = tf.keras
B = K.backend
L = K.layers

CONCAT_AXIS = -1
FN = "tanh"


def use_ragged_actuator(agent, parts, reuse=False):
    brick_type = "ragged_actuator"
    if brick_type not in id:
        id = make_id([id, brick_type], reuse=reuse)

    d_out = parts["d_out"]
    d_in = agent.code_spec.size

    units = d_in ** 2 + 1  # add one to account for the atom
    layer_list = [(units, FN), (units, FN), (d_out, FN)]
    mlp_parts = dict(
        brick_type="mlp", id=id, layer_list=layer_list, input_shape=(units,))
    mlp = agent.pull_brick(mlp_parts, result="brick")

    def call(code, coords, d_out):
        x = tf.map_fn(
                lambda coord: mlp(
                        tf.concat([code, coord], CONCAT_AXIS)
                    ),
                coords)
        x = agent.pull_brick(agent, f"{d_out}", x, result="out", reuse=True)
        return x

    parts = dict(
        brick_type=brick_type, id=id, input=input, mlp=mlp, call=call)

    return parts

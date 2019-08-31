# ragged_actuator.py - bion
# decode sequences as simply as possible

import tensorflow as tf

from nature import MLP, Brick

K = tf.keras
B, L = K.backend, K.layers

CONCAT_AXIS = -1
FN = "tanh"


def RaggedActuator(agent, spec):
    d_in = agent.code_spec.size
    d_out = spec.shape[-1]

    units = d_in ** 2 + 1  # add one to account for the atom
    units_list = [units, units, d_out]
    fn_list = [FN, FN, FN]
    mlp = MLP(units_list=units_list, fn_list=fn_list)

    def call(self, code, coords):
        return tf.map_fn(
                lambda coord: mlp(
                        tf.concat([code, coord], CONCAT_AXIS)
                    ),
                coords)
    return Brick(spec, d_in, d_out, units, units_list, fn_list, mlp, agent)

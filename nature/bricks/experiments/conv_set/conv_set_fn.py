# ragged_sensor.py - bion
# encode sequences as simply as possible

import tensorflow as tf

from tools import map_sets_tf
from nature import MLP, Brick

K = tf.keras
B, L = K.backend, K.layers

RESIZE_MATRIX_STDDEV = 0.04
CONCATENATE_ON_INDEX = -1
POOL_ON_INDEX = 1
MAX_SET_SIZE = 2
FN = "tanh"


def ConvSet(agent, in_spec, out_spec):

    d_out = agent.code_spec.size
    mlps = []
    for set_size in reversed(range(MAX_SET_SIZE)):
        units = d_out * (set_size + 1)  # add one to account for the atom
        units_list = [units, units, d_out]
        fns_list = [FN, FN, FN]
        mlp = MLP(agent, units_list=units_list, fns_list=fns_list)
        mlps.append((set_size, mlp))

    def call_mlp(set, atom):
        concatenated_atoms = tf.concat([*set, atom], CONCATENATE_ON_INDEX)
        return mlp(concatenated_atoms)

    def pool_atoms(atoms):
        return tf.reduce_sum(atoms, POOL_ON_INDEX)

    def set_conv(x, set_size, mlp):
        return tf.map_fn(
            lambda atom: map_sets_tf(
                x, set_size, lambda set: call_mlp(set, atom), pool_atoms))

    def call(self, x):
        d_in = tf.shape(x)[-1]
        if d_in is not d_out:
            resizer = tf.random.truncated_normal(
                (d_in, d_out), mean=1., stddev=RESIZE_MATRIX_STDDEV)
            x = tf.map_fn(lambda item: B.dot(item, resizer), x)
        for set_size, mlp in mlps:
            x = set_conv(x, set_size, mlp)
        return pool_atoms(x)
    return Brick(
        agent, in_spec, out_spec, d_out, mlps, call_mlp, pool_atoms, set_conv,
        call)

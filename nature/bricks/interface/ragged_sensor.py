import tensorflow as tf

from nature import use_mlp

L = tf.keras.layers

FN = "tanh"


def use_ragged_sensor(agent, spec):
    """args: agent, spec"""
    out_spec = agent.code_spec

    units = spec.size ** 2 + 1
    units_list = [units, units, agent.code_spec.size]
    fn_list = [FN, FN, FN]
    mlp = use_mlp(units_list=units_list, fn_list=fn_list)
    reshape = L.Reshape(out_spec.shape)

    def fn(accumulator, item):
        return accumulator + mlp(item)

    def call(x):
        y = tf.foldl(fn, x)
        return reshape(y)
    return call

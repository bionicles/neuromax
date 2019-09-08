import tensorflow as tf

from nature import MLP, Brick

L = tf.keras.layers

FN = "tanh"


def RaggedSensor(agent, spec):
    """args: agent, spec"""
    out_spec = agent.code_spec

    units = spec.size ** 2 + 1
    units_list = [units, units, agent.code_spec.size]
    fn_list = [FN, FN, FN]
    mlp = MLP(units_list=units_list, fn_list=fn_list)
    reshape = L.Reshape(out_spec.shape)

    def fn(accumulator, item):
        return accumulator + mlp(item)

    def call(self, x):
        y = tf.foldl(fn, x)
        return reshape(y)
    return Brick(
        agent, spec, out_spec, units, units_list,
        fn_list, mlp, reshape, fn, call)

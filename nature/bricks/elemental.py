import tensorflow as tf

from nature import use_dense

K = tf.keras
L = K.layers

UNITS = 128


def use_add(agent, parts):
    assert "units" in parts.keys()
    parts.dense = dense = use_dense(parts.units)
    parts.adder = adder = L.Add()

    def call(x):
        out = dense(x)
        return adder([x, out])
    parts.call = call
    return parts


def use_multiply(agent, parts):
    if "units" not in parts.keys():
        parts.units = UNITS
    parts.dense1 = dense1 = use_dense(parts.units)
    parts.dense2 = dense2 = use_dense(parts.units)
    parts.multiplier = multiplier = L.Multiply()

    def call(x):
        o1 = dense1(x)
        o2 = dense2(x)
        return multiplier([o1, o2])
    parts.call = call
    return parts

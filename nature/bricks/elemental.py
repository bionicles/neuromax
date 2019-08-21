import tensorflow as tf

from nature import use_dense, use_layer
from tools import make_uuid

K = tf.keras
L = K.layers

LAYER_FN = use_dense
UNITS = 128


def use_add(agent, id, input, layer_fn=LAYER_FN, return_brick=False):
    id = make_uuid([id, "add"])

    d_in = input.shape[-1]
    _, op = use_layer(agent, id, layer_fn, input, d_in)
    adder = L.Add()
    parts = dict(op=op, adder=adder)

    def call(x):
        out = op(x)
        return adder([x, out])
    return agent.pull_brick(id, parts, call, input, return_brick)


def use_multiply(
        agent, id, input, layer_fn=LAYER_FN, units=UNITS, return_brick=False):
    id = make_uuid([id, "multiply"])

    layer1 = use_layer(agent, id, layer_fn, input, units)
    layer2 = use_layer(agent, id, layer_fn, input, units)
    multiplier = L.Multiply()
    parts = dict(layer1=layer1, layer2=layer2, multiplier=multiplier)

    def call(x):
        o1 = layer1(x)
        o2 = layer2(x)
        return multiplier([o1, o2])
    return agent.pull_brick(id, parts, call, input, return_brick)

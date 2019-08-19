import tensorflow as tf

from .layers.get_layer import get_layer
from .layers.dense import get_dense_out
from tools.get_brick import get_brick

K = tf.keras
L = K.layers

LAYER_FN = get_dense_out
UNITS = 128


def get_add_out(agent, id, input, layer_fn=LAYER_FN, return_brick=False):
    d_in = input.shape[-1]
    _, op = get_layer(agent, id, input, layer_fn, d_in)
    adder = L.Add()

    def add(input):
        out = op(input)
        return adder([input, out])
    return get_brick(add, input, return_brick)


def get_multiply_out(
        agent, id, input, layer_fn=LAYER_FN, units=UNITS, return_brick=False):
    if "multiply" not in id:
        id = f"{id}_multiply"

    layer1 = get_layer(agent, id, input, layer_fn, units)
    layer2 = get_layer(agent, id, input, layer_fn, units)
    multiplier = L.Multiply()

    def multiply(input):
        o1 = layer1(input)
        o2 = layer2(input)
        return multiplier([o1, o2])
    return get_brick(multiply, input, return_brick)

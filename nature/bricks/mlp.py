import tensorflow as tf

from .layers.dense import get_dense_out
from .helpers.activations import swish
from tools.get_brick import get_brick

UNITS, FN, LAYERS = 256, swish, 2


def get_mlp_out(
        agent, id, out, layer_list=None, last_layer=None, return_brick=False):
    if layer_list is None:
        layer_list = [(UNITS, FN) for _ in range(LAYERS)]
        if last_layer is not None:
            layer_list[-1] = last_layer

    mlp = []
    for units, fn in layer_list:
        _, layer = get_dense_out(
            agent, id, out, units=units, fn=fn, return_brick=True)
        mlp.append(layer)

    def use_mlp(out):
        for layer in mlp:
            out = layer(out)
        return out
    return get_brick(use_mlp, out, return_brick)

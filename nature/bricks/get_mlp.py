import tensorflow as tf

from .activations import swish
from .dense import get_dense

K = tf.keras

UNITS, FN, LAYERS = 256, swish, 2


def get_mlp(agent, brick_id, input_shape, layer_list=None, last_layer=None):
    if layer_list is None:
        if last_layer is None:
            layer_list = [(UNITS, FN) for _ in range(LAYERS)]
        else:
            layer_list = [(UNITS, FN) for _ in range(LAYERS - 1)]
            layer_list.append(last_layer)
    layers = [K.Input(input_shape)] + [
        get_dense(agent, brick_id, units=units, fn=fn)
        for units, fn in layer_list]
    return K.Sequential(layers)

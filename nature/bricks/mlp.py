import tensorflow as tf

from nature import use_linear, use_fn
K = tf.keras

LAYERS, UNITS, FN = 2, 256, None
UNITS = [UNITS for _ in range(LAYERS)]
FNS = [FN for _ in range(LAYERS)]


def use_mlp(units=UNITS, fns=FNS):
    """
    make a sequential MLP... default is [256, 256] with no activation
    Args:
        tensor_or_shape: tensor, or tuple
    kwargs:
        units: list of int for units
        fns: list of strings for activations (default None)
    """
    layers = []
    for i, units in enumerate(units):
        dense = use_linear(units)
        layers.append(dense)
        if fns[i]:
            layers.append(use_fn(fns[i]))

    def call(x):
        y = x
        for layer in layers:
            y = layer(y)
        return y
    return call

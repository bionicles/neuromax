import tensorflow as tf

from nature import Linear, Fn, Brick

K = tf.keras

LAYERS, UNITS, FN = 2, 256, None
UNITS_LIST = [UNITS for _ in range(LAYERS)]
FNS_LIST = [FN for _ in range(LAYERS)]


def MLP(agent, units_list=UNITS_LIST, fns_list=FNS_LIST):
    """
    make a sequential MLP... default is [256, 256] with no activation
    Args:
        tensor_or_shape: tensor, or tuple
    kwargs:
        units_list: list of int for units
        fns_list: list of strings for activations (default None)
    """
    layers = []
    for i, units in enumerate(units_list):
        dense = Linear(units=units)
        layers.append(dense)
        if fns_list[i]:
            layers.append(Fn(fns_list[i]))

    def call(self, x):
        for layer in layers:
            x = layer(x)
        return x
    return Brick(units_list, fns_list, layers, call, agent)

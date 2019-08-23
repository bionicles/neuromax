import tensorflow as tf

from nature import use_linear, use_input, use_fn
K = tf.keras

LAYERS, UNITS, FN = 2, 256, None
UNITS_LIST = [UNITS for _ in range(LAYERS)]
FN_LIST = [UNITS for _ in range(LAYERS)]

BRICK_TYPE = "mlp"


def use_mlp(tensor_or_shape, units_list=UNITS_LIST, fn_list=FN_LIST):
    """
    make a sequential MLP... default is [256, 256] with no activation
    Args:
        tensor_or_shape: tensor, or tuple
    kwargs:
        units_list: list of int for units
        fn_list: list of strings for activations (default none)
    """
    layers = [use_input(tensor_or_shape)]
    for i, units in enumerate(units_list):
        dense = use_linear(units)
        layers.append(dense)
        if fn_list[i]:
            layers.append(use_fn(fn_list[i]))
    mlp = K.Sequential(layers)
    return mlp

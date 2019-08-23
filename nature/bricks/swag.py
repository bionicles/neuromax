import tensorflow as tf

from nature import use_linear, use_fn

K = tf.keras
L = K.layers

CONCATENATION_AXIS = -1
POWER = 3
UNITS = 32
FN = None


def use_swag(
        power=POWER, units=UNITS,
        fn=None, out_fn=None, out_units=UNITS, reshape=None):
    """
    apply POWER linear layers with UNITS and multiply with last value
    then flatten and mix the outputs with a linear layer with OUT_UNITS

    kwargs:
        power: int for the number of layers to multiply (default 3)
        units: int for the number of units per linear layer (default 32)
        fn: string name of the activation to use (default None)
        out_units: int for the units in the final mixer (default 32)
        out_fn: last activation (default None)
    """
    layers = [use_linear(units) for _ in range(power)]
    if fn:
        fn = use_fn(fn)
    multiply = L.Multiply()
    flatten = L.Flatten()
    concat = L.Concatenate(CONCATENATION_AXIS)
    output_layer = use_linear(units)
    if out_fn:
        out_fn = use_fn(out_fn)
    if reshape:
        reshape = L.Reshape(reshape)

    def call(x):
        y = tf.identity(x)
        ys = []
        for i, layer in enumerate(layers):
            y = layer(y)
            if fn:
                y = fn(y)
            if len(ys) > 0:
                y = multiply([ys[-1], y])
            ys.append(y)
        ys = [flatten(o) for o in ys]
        y = concat(ys)
        y = output_layer(y)
        if out_fn:
            y = out_fn(y)
        if reshape:
            y = reshape(y)
        return y
    return call

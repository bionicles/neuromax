import tensorflow as tf

from nature import use_linear, use_fn

K = tf.keras
L = K.layers

CONCATENATION_AXIS = -1
POWER = 3
UNITS = 32
FN = None


def use_swag(power=POWER, units=UNITS, fn=None, out_fn=None, out_units=UNITS):
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

    def call(x):
        xs = []
        for i, layer in enumerate(layers):
            x = layer(x)
            if fn:
                x = fn(x)
            if len(xs) > 0:
                x = multiply([xs[-1], x])
            xs.append(x)
        xs = [flatten(o) for o in xs]
        x = concat(xs)
        x = output_layer(x)
        if out_fn:
            x = out_fn(x)
        return x
    return call

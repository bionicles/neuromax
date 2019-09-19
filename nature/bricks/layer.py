import nature

LAYER_FN = nature.NoiseDrop
UNITS = 8

def Layer(units=UNITS, layer_fn=LAYER_FN):
    """
    get a layer with a desired output dimension

    units_or_filters: int for linear units or dense filters
    layer_fn: callable to build the layer
    """
    return layer_fn(units=units)

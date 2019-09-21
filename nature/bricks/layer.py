import nature

LAYER_FN = nature.NoiseDrop
UNITS = 8

def Layer(units=UNITS, layer_fn=LAYER_FN):
    """
    units: int
    layer_fn: callable to build the layer
    """
    return layer_fn(units=units)

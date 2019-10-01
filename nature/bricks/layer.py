import nature

LAYERS = [nature.FC, nature.NoiseDrop]
UNITS = [32, 64, 128, 256, 512]


def Layer(AI, units=None, layer_fn=None, hyper=None):
    if layer_fn is nature.Layer:
        layer_fn = None
    if not layer_fn:
        layer_fn = AI.pull("layer_fn", LAYERS, id=False)
    if not units:
        units = AI.pull("units", UNITS, id=False)
    layer = layer_fn(units=units)
    return layer

import nature

LAYERS = ["FC", "NoiseDrop"]
UNITS = [32, 64, 128, 256, 512, 1024]


def Layer(AI, units=None, layer_fn=None, hyper=None):
    if layer_fn is nature.Layer:
        layer_fn = None
    if not layer_fn:
        layer_fn = getattr(nature, AI.pull("layer_fn", LAYERS))
    if not units:
        units = AI.pull("units", UNITS)
    return layer_fn(units=units)

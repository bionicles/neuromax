import tensorflow as tf
import nature

LAYER_FN = nature.FC
UNITS = 8


def Layer(units=UNITS, layer_fn=LAYER_FN):
    return layer_fn(units=units)

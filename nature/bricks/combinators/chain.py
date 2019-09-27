from random import choice
import tensorflow as tf

from tools import safe_sample
import nature

OPTIONS = [nature.ResBlock]
MIN, MAX = 2, 2
UNITS = 3

L = tf.keras.layers


def Chain(units=UNITS, constructor=OPTIONS):
    if isinstance(constructor, list):
        constructor = choice(OPTIONS)
    n_bricks = safe_sample(MIN, MAX)
    wrapped = constructor
    if constructor is nature.Recirculator:
        def wrapped():
            return constructor(units=units)

    def call(x):
        arr = []
        y = x
        for brick_number in range(n_bricks):
            y = wrapped()(y)
            arr.append(y)
        return L.Concatenate(1)(arr)
    return call

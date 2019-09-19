import tensorflow as tf

import nature

K = tf.keras
L = K.layers

LAYER = nature.Layer
UNITS = 32
POWER = 4


class SWAG(L.Layer):
    def __init__(self, power=POWER, layer_fn=LAYER, units=UNITS):
        super(SWAG, self).__init__()
        self.layers = []
        for p in range(power):
            norm_preact = nature.NormPreact()
            super(SWAG, self).__setattr__(f"np_{p}", norm_preact)
            layer = layer_fn(units=units)
            super(SWAG, self).__setattr__(f"layer_{p}", layer)
            self.layers.append((norm_preact, layer))
        self.multiply = L.Multiply()
        self.norm = nature.Norm()
        self.add = L.Add()
        self.built = True

    @tf.function
    def call(self, x):
        ys = []
        y = x
        for norm_preact, layer in self.layers:
            y = norm_preact(y)
            y = layer(y)
            if len(ys) > 0:
                y = self.multiply([ys[-1], y])
            ys.append(y)
        y = self.add(ys)
        y = self.norm(y)
        return y

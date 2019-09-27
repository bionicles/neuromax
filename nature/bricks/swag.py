import tensorflow as tf
import nature

L = tf.keras.layers
LAYER = nature.Layer
NORM = nature.Norm
KEY = 'mish'
UNITS = 4
POWER = 5


class SWAG(L.Layer):

    def __init__(self, power=POWER, layer_fn=LAYER, units=UNITS):
        super(SWAG, self).__init__()
        self.layers = []
        for p in range(power):
            norm_preact = nature.NormPreact(key=KEY)
            super().__setattr__(f"np_{p}", norm_preact)
            layer = layer_fn(units=units)
            super().__setattr__(f"layer_{p}", layer)
            self.layers.append((norm_preact, layer))
        self.multiply = L.Multiply()
        self.addnorm = nature.AddNorm()
        self.built = True

    @tf.function
    def call(self, x):
        ys = [x]
        for norm_preact, layer in self.layers:
            x = ys[-1]
            y = norm_preact(x)
            y = layer(y)
            y = self.multiply([x, y])
            ys.append(y)
        return self.addnorm(ys)

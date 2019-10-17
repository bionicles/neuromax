# The SWAG Algorithm (loosely based on)
# https://arxiv.org/abs/1811.11813

# also similar to a N-ary generalization of:
# Universal Approximation with Quadratic Deep Networks
# https://arxiv.org/pdf/1808.00098.pdf

import tensorflow as tf
import nature

L = tf.keras.layers
LAYER = nature.Layer
MIN_POWER, MAX_POWER = 2, 8


class SWAG(L.Layer):

    def __init__(self, AI, layer_fn=LAYER, units=None):
        super(SWAG, self).__init__()
        power = AI.pull("swag_power", MIN_POWER, MAX_POWER)
        self.zero = LAYER(AI, units=units)
        self.fn = nature.Fn(AI)
        self.layers = []
        for p in range(power):
            np = nature.NormPreact(AI)
            super().__setattr__(f"np_{p}", np)
            one = LAYER(AI, units=units)
            super().__setattr__(f"one_{p}", one)
            self.layers.append((np, one))
        self.addnorm = nature.AddNorm()
        self.built = True

    @tf.function
    def call(self, x):
        ys = [self.zero(self.fn(x))]
        for np, one in self.layers:
            x = np(ys[-1])
            ys.append(x * one(x))
        return self.addnorm(ys)

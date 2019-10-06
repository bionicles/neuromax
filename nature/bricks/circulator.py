import tensorflow as tf
from tools import error
import nature

L = tf.keras.layers
LAYER_OPTIONS = ['Layer', 'Attention', 'MLP', 'SWAG']
LOOP_OPTIONS = [1, 2, 3, 4]


class Circulator(L.Layer):

    def __init__(self, AI, units=None, layer_fn=None):
        super().__init__()
        if not layer_fn:
            layer = AI.pull("circulator_layer", LAYER_OPTIONS, id=False)
            self.layer = getattr(nature, layer)
        self.n_loops = AI.pull("circulator_loops", LOOP_OPTIONS, id=False)
        self.ai = AI

    def build(self, shape):
        self.code = self.add_weight(
            "code", shape,
            initializer=nature.Init(), regularizer=nature.L1L2())
        self.encode = self.layer(self.ai, units=shape[-1])
        self.decode = self.layer(self.ai, units=shape[-1])
        self.fn = nature.Fn(self.ai)
        self.out = L.Add()
        super().build(shape)

    @tf.function
    def call(self, x):
        prev_prediction = prev_reconstruction = x
        prev_code = self.code
        errors = []
        for n in range(self.n_loops):
            prediction = self.fn(self.decode(prev_code))
            code = self.fn(self.encode(prev_reconstruction))
            reconstruction = self.fn(self.decode(code))
            errors.extend([
                error(prev_prediction, prediction),
                error(x, prediction) * 420.,
                error(prev_reconstruction, reconstruction),
                error(x, reconstruction) * 420.,
                error(prev_code, code)])
            prev_reconstruction = reconstruction
            prev_prediction = prediction
            prev_code = code
        y = self.out(errors)
        return y

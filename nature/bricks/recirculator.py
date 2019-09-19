import tensorflow as tf
import nature

L = tf.keras.layers
LAYER_FN = nature.Layer
N_LOOPS = 2
UNITS = 8

class Recirculator(L.Layer):

    def __init__(self, units=UNITS, layer_fn=LAYER_FN):
        super(Recirculator, self).__init__()
        self.layer_fn = layer_fn
        self.units = units

    def build(self, shape):
        self.encode = self.layer_fn(units=self.units)
        self.decode = self.layer_fn(units=shape[-1])
        self.addnorm = nature.AddNorm()
        self.built = True

    @tf.function
    def loop(self, input, target):
        reconstruction = input
        errors = []
        for _ in range(N_LOOPS):
            code = self.encode(reconstruction)
            reconstruction = self.decode(code)
            error = tf.math.abs(input - reconstruction)
            errors.append(error)
            if input is target:
                continue
            target_error = tf.math.abs(target - reconstruction)
            errors.append(target_error)
        return errors

    @tf.function
    def call(self, x):
        outputs = []
        noise = tf.random.normal(tf.shape(x))
        errors = self.loop(noise, x)
        errors = errors + self.loop(x, x)
        for error in errors:
            self.add_loss(error)
        y = self.addnorm(errors)
        return y

import tensorflow as tf
import nature

LAYER_FN = nature.Layer
N_LOOPS, UNITS = 4, 16


class Recirculator(tf.keras.layers.Layer):

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
        shape = tf.shape(input)
        prior_code = tf.random.normal((shape[0], shape[1], self.units))
        reconstruction = input
        errors = []
        for n in range(N_LOOPS):
            code = self.encode(reconstruction)
            if n > 0:
                error = tf.math.abs(code - prior_code)
                errors.append(error)
            prior_code = code
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
        noise = tf.random.normal(tf.shape(x))
        errors = self.loop(noise, x) + self.loop(x, x)
        for error in errors:
            self.add_loss(error)
        y = self.addnorm(errors)
        return y

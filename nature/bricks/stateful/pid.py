import tensorflow as tf
import nature
L = tf.keras.layers


class PID(L.Layer):

    def __init__(self, *args, **kwargs):
        super(PID, self).__init__()

    def build(self, shape):
        self.mixer = nature.FC(units=shape[-1], activation="gelu")
        self.derivative = nature.Derivative()
        self.integral = nature.Integral()
        self.built = True

    @tf.function
    def call(self, p):
        i = self.integral(p)
        d = self.derivative(p)
        pid = tf.concat([p, i, d], -1)
        return self.mixer(pid)

    def compute_output_shape(self, shape):
        return shape

import tensorflow as tf
from tools import get_size
import nature
L = tf.keras.layers


class PID(L.Layer):

    def __init__(self, *args, **kwargs):
        super(PID, self).__init__()

    def build(self, shape):
        size = get_size(shape[1:])
        self.mixer = nature.FC(units=size)
        self.derivative = nature.Derivative()
        self.integral = nature.Integral()
        self.shape = shape
        self.built = True

    @tf.function
    def call(self, p):
        i = self.integral(p)
        d = self.derivative(p)
        pid = tf.concat([p, i, d], -1)
        return self.mixer(pid)

    def compute_output_shape(self, shape):
        return shape

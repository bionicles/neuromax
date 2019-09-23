import tensorflow as tf
from tools import log
import nature

L = tf.keras.layers
NORM = L.BatchNormalization


class Sensor(L.Layer):
    def __init__(self, AI, spec):
        super(Sensor, self).__init__()
        self.d_code = AI.code_spec.shape[-1]
        self.in_spec = spec
        self.AI = AI

    def build(self, shape):
        self.channel_changer = self.norm_2 = tf.identity
        self.expander = tf.identity
        if len(shape) is 2:
            self.expander = L.Lambda(lambda x: tf.expand_dims(x, -1))
            shape = tuple([*shape, 1])
        new_dimension = shape[-1] + len(shape[1:-1])
        self.dense = tf.identity
        if len(shape) is 4:
            self.dense = nature.DenseBlock(layer_fn=nature.Conv2D)
            new_dimension = new_dimension + 8
        if new_dimension is not self.d_code:
            self.channel_changer = nature.OP_1D(units=self.d_code)
            self.norm_2 = NORM()
        self.reshape = L.Reshape((-1, new_dimension))
        self.coords = nature.Coordinator(shape)
        self.norm_1 = NORM()
        self.built = True

    def call(self, x):
        x = self.expander(x)
        x = self.coords(x)
        x = self.dense(x)
        x = self.reshape(x)
        x = self.norm_1(x)
        x = self.channel_changer(x)
        x = self.norm_2(x)
        return x

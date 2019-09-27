import tensorflow as tf
# from tools import log
import nature

L = tf.keras.layers


class Sensor(L.Layer):
    def __init__(self, AI, spec):
        super(Sensor, self).__init__()
        self.d_code = AI.code_spec.shape[-1]
        self.in_spec = spec
        self.AI = AI

    def build(self, shape):
        self.channel_changer = self.expander = self.dense = tf.identity
        if len(shape) is 2:
            self.coords = tf.identity
            self.expander = L.Lambda(lambda x: tf.expand_dims(x, -1))
            shape = tuple([*shape, 1])
            new_dimension = shape[-1]
        elif len(shape) is 3 or 4:
            self.coords = nature.Coordinator(shape)
            new_dimension = shape[-1] + len(shape[1:-1])
        if len(shape) is 4:
            self.dense = nature.DenseBlock(layer_fn=nature.Conv2D)
            new_dimension = new_dimension + self.dense.d_increase
        if new_dimension is not self.d_code:
            self.channel_changer = nature.OP_1D(units=self.d_code)
        self.reshape = L.Reshape((-1, new_dimension))
        super().build(shape)

    @tf.function
    def call(self, x):
        x = self.expander(x)
        x = self.coords(x)
        x = self.dense(x)
        x = self.reshape(x)
        x = self.channel_changer(x)
        return x

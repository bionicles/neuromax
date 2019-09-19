import tensorflow as tf

from nature import Coordinator, Norm, OP_1D

L = tf.keras.layers


class Sensor(L.Layer):
    def __init__(self, d_code):
        super(Sensor, self).__init__()
        self.d_code = d_code

    def build(self, shape):
        self.channel_changer = self.norm_2 = tf.identity
        new_dimension = shape[-1] + len(shape[1:-1])
        if new_dimension is not self.d_code:
            self.channel_changer = OP_1D(units=self.d_code)
            self.norm_2 = Norm()
        self.reshape = L.Reshape((-1, new_dimension))
        self.coords = Coordinator(shape)
        self.norm_1 = Norm()
        self.built = True

    @tf.function
    def call(self, x):
        x = self.coords(x)
        x = self.reshape(x)
        x = self.norm_1(x)
        x = self.channel_changer(x)
        x = self.norm_2(x)
        return x

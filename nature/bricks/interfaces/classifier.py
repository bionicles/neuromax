import tensorflow as tf
import nature

L = tf.keras.layers
# NORM = L.BatchNormalization
LAYER = nature.FC
FN1 = 'mish'
FN2 = 'softmax'


class Classifier(L.Layer):
    def __init__(self, shape):
        super(Classifier, self).__init__()
        self.out_shape = shape

    def build(self, shape):
        self.one = nature.MLP()
        # self.norm = NORM()
        self.resize = nature.Resizer(self.out_shape, layer=LAYER, key=FN1)
        self.out = LAYER(units=self.out_shape[-1], activation=FN2)
        super().build(shape)

    @tf.function
    def call(self, x):
        x = self.one(x)
        # x = self.norm(x)
        x = self.resize(x)
        x = x - tf.math.reduce_max(x, axis=1, keepdims=True)
        x = x / tf.math.reduce_sum(x, axis=1, keepdims=True)
        x = self.out(x)
        return x

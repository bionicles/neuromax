import tensorflow as tf
import nature

L = tf.keras.layers
FIRST_OPTIONS = ["SWAG", "MLP", "Attention", "Delta"]
LAYER = nature.FC
FN2 = 'softmax'


class Classifier(L.Layer):
    def __init__(self, AI, shape):
        first = AI.pull("classifier_first", FIRST_OPTIONS)
        super().__init__(name=f"{shape[-1]}C_{first}_classifier")
        self.first = getattr(nature, first)
        self.out_shape = shape
        self.ai = AI

    def build(self, shape):
        self.one = self.first(self.ai)
        self.resize = nature.Resizer(self.ai, self.out_shape, key=FN2)
        super().build(shape)

    @tf.function
    def call(self, x):
        x = self.one(x)
        x = x - tf.math.reduce_max(x, axis=1, keepdims=True)
        x = x / tf.math.reduce_sum(x, axis=1, keepdims=True)
        x = self.resize(x)
        return x

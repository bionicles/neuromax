import tensorflow as tf
import nature

L = tf.keras.layers
FIRST_OPTIONS = [nature.SWAG, nature.MLP, nature.Attention]
LAYER = nature.FC
FN2 = 'softmax'


class Classifier(L.Layer):
    def __init__(self, AI, shape):
        super(Classifier, self).__init__()
        self.out_shape = shape
        self.ai = AI

    def build(self, shape):
        first = self.ai.pull("classifier_first_layer", FIRST_OPTIONS, id=False)
        self.one = first(self.ai, layer_fn=LAYER)
        self.resize = nature.Resizer(
            self.ai, self.out_shape, layer=LAYER, key=FN2)
        super().build(shape)

    @tf.function
    def call(self, x):
        x = self.one(x)
        x = x - tf.math.reduce_max(x, axis=1, keepdims=True)
        x = x / tf.math.reduce_sum(x, axis=1, keepdims=True)
        x = self.resize(x)
        return x

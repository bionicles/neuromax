import tensorflow as tf
import nature

L = tf.keras.layers
FIRST_OPTIONS = [nature.SWAG, nature.MLP, nature.Attention]
SIZE_OPTIONS = [1, 2, 3, 4, 5]
LAYER = nature.FC
FN = None


class Regresser(L.Layer):

    def __init__(self, AI, out_shape):
        super(Regresser, self).__init__()
        self.ensemble_size = AI.pull(
            'regresser_ensemble_size', SIZE_OPTIONS, id=False)
        self.out_shape = out_shape
        self.ai = AI

    def build(self, shape):
        first = self.ai.pull("regresser_first_layer", FIRST_OPTIONS, id=False)
        self.one = first(self.ai, layer_fn=LAYER)
        ensemble_shape = list(self.out_shape) + [self.ensemble_size]
        self.resize = nature.Resizer(
            self.ai, ensemble_shape, layer=LAYER, key=FN)
        super().build(shape)

    @tf.function
    def call(self, x):
        x = self.one(x)
        x = self.resize(x)
        return tf.reduce_mean(x, axis=-1)

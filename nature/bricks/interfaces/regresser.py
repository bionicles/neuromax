import tensorflow as tf
from tools import make_id
import nature

L = tf.keras.layers
FIRST_OPTIONS = ["SWAG", "MLP", "Attention"]
SIZE_OPTIONS = [1, 2, 3, 4, 5]
LAYER = nature.FC
FN = "identity"


class Regresser(L.Layer):

    def __init__(self, AI, out_shape):
        self.ensemble_size = AI.pull('regresser_ensemble_size', SIZE_OPTIONS)
        first = AI.pull("regresser_first_layer", FIRST_OPTIONS)
        self.first = getattr(nature, first)
        super(Regresser, self).__init__(
            name=make_id(f"regresser_{self.ensemble_size}_{first}"))
        self.out_shape = out_shape
        self.ai = AI

    def build(self, shape):
        self.one = self.first(self.ai, layer_fn=LAYER)
        ensemble_shape = list(self.out_shape) + [self.ensemble_size]
        self.resize = nature.Resizer(self.ai, ensemble_shape, key=FN)
        super().build(shape)

    @tf.function
    def call(self, x):
        x = self.one(x)
        x = self.resize(x)
        return tf.reduce_mean(x, axis=-1)

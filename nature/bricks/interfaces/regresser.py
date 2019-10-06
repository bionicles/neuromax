import tensorflow as tf
from tools import make_id
import nature

L = tf.keras.layers
FIRST_OPTIONS = ["SWAG", "MLP"]
SIZE_OPTIONS = [1, 2, 3, 4, 5]
FN = "identity"


class Regresser(L.Layer):

    def __init__(self, AI, out_shape):
        self.ensemble_size = AI.pull('regresser_ensemble_size', SIZE_OPTIONS)
        first = AI.pull("regresser_first", FIRST_OPTIONS)
        self.first = getattr(nature, first)
        super(Regresser, self).__init__(
            name=make_id(f"{self.ensemble_size}X_{first}_regresser"))
        self.out_shape = out_shape
        self.ai = AI

    def build(self, shape):
        ensemble_shape = list(self.out_shape) + [self.ensemble_size]
        self.resize = nature.Resizer(self.ai, ensemble_shape, key=FN)
        self.one = self.first(self.ai)
        super().build(shape)

    @tf.function
    def call(self, x):
        return tf.reduce_mean(self.resize(self.one(x)), axis=-1)

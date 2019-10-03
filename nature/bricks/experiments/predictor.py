import tensorflow as tf

from nature import Resizer, ResBlock, AllAttention
from tools import safe_sample

K, L = tf.keras, tf.keras.layers

MIN_BLOCKS, MAX_BLOCKS = 4, 4

class Predictor(L.Layer):

    def __init__(self, out_shape=None):
        super(Predictor, self).__init__()
        self.out_shape = out_shape

    def build(self, shape):
        self.blocks = []
        for n in range(safe_sample(MIN_BLOCKS, MAX_BLOCKS)):
            block = ResBlock(layer_fn=AllAttention)
            setattr(self, f"block_{n}", block)
            self.blocks.append(block)
        self.add = L.Add()
        self.built = True

    def call(self, x):
        array = []
        p = tf.random.truncated_normal(tf.shape(x))
        for block in self.blocks:
            p = block(p)
            surprise = tf.math.abs(x - p)
            self.add_loss(surprise)
            array.append(surprise)
        return surprise

    def compute_output_shape(self, shape):
        return self.out_spec

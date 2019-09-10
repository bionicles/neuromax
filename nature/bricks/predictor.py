import tensorflow as tf

from nature import Resizer, Coordinator, Norm, ResBlock, AllAttention
from tools import safe_sample

K = tf.keras
L = K.layers

MIN_BLOCKS, MAX_BLOCKS = 1, 4

class Predictor(L.Layer):
    """generates inputs and returns a surprise value"""

    def __init__(self, out_shape=None):
        super(Predictor, self).__init__()
        self.out_shape = out_shape
        self.resize = None

    def build(self, shape):
        self.coordinator = Coordinator(shape)
        self.x_norm = Norm()
        self.blocks = []
        for n in range(safe_sample(MIN_BLOCKS, MAX_BLOCKS)):
            block = ResBlock(layer_fn=AllAttention)
            setattr(self, f"block_{n}", block)
            self.blocks.append(block)
        if self.out_shape:
            self.resize = Resizer(self.out_shape)
        self.built = True

    def call(self, x):
        p = tf.random.truncated_normal(tf.shape(x))
        p = self.coordinator(p)
        x = self.x_norm(x)
        x = self.coordinator(x)
        for block in self.blocks:
            p = tf.reshape(block(p), tf.shape(x))
            surprise = tf.losses.MSLE(x, p)
            self.add_loss(surprise)
        if self.resize:
            surprise = self.resize(surprise)
        return surprise

    def compute_output_shape(self, shape):
        return self.out_spec

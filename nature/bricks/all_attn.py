import tensorflow as tf
import nature

L = tf.keras.layers

LAYER = nature.Quadratic
MEMORY_SIZE = 512
D_MODEL = 4

class AllAttention(L.Layer):
    def __init__(self, keepdim=True):
        super(AllAttention, self).__init__()
        self.keepdim = keepdim

    def build(self, shape):
        self.batch_size = shape[0]
        d_in = shape[-1]
        self.wq = LAYER(units=D_MODEL)
        self.wk = LAYER(units=D_MODEL)
        if self.keepdim:
            self.out = LAYER(units=d_in)
        else:
            self.out = tf.identity
        self.attention = L.Attention()
        self.reshape = L.Reshape((-1, d_in))
        self.concat = L.Concatenate(1)
        self.memory = self.add_weight(
                shape=(1, MEMORY_SIZE, D_MODEL), initializer='glorot_normal',
                regularizer=nature.L1L2(), trainable=True)
        self.built = True

    @tf.function
    def call(self, x):
        x = self.reshape(x)
        seq_len = tf.shape(x)[1]
        memory = tf.tile(self.memory, [self.batch_size, 1, 1])
        q, k = self.wq(x), self.wk(x)
        q = self.concat([q, memory])
        k = self.concat([k, memory])
        attended = self.attention([q, k])
        y, _ = tf.split(attended, [seq_len, MEMORY_SIZE], axis=1)
        y = self.out(y)
        return y

    def compute_output_shape(self, shape):
        return shape

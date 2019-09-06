import tensorflow as tf

from tools import tile_to_batch_size
from nature import Linear

L = tf.keras.layers

MEMORY_SIZE = 4


class AllAttention(L.Layer):
    def __init__(self):
        super(AllAttention, self).__init__()

    def build(self, input_shape):
        self.d_in = input_shape[-1]
        self.wq, self.wk = Linear(units=self.d_in), Linear(units=self.d_in)
        self.attention = L.Attention()
        self.memory = self.add_weight(
                shape=(MEMORY_SIZE, self.d_in),
                initializer='random_normal', trainable=True)
        self.built = True

    def call(self, x):
        rank = len(tf.shape(x))
        batch_size = tf.shape(x)[0]
        if rank is 2:
            x = tf.expand_dims(x, 1)
        elif rank is 4:
            x = tf.reshape(x, [batch_size, -1, self.d_in])
        seq_len = tf.shape(x)[1]
        memory = self.memory
        if memory.shape[0] is not x.shape[0]:
            memory = tile_to_batch_size(batch_size, memory)
        x = tf.concat([x, memory], 1)
        attended = self.attention([self.wq(x), self.wk(x)])
        y, self.memory = tf.split(attended, [seq_len, MEMORY_SIZE], axis=1)
        return y

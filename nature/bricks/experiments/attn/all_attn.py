import tensorflow as tf

from nature import use_linear

L = tf.keras.layers

MEMORY_SIZE = 4


class AllAttention(L.Layer):
    def __init__(self):
        super(AllAttention, self).__init__()

    def build(self, input_shape):
        batch_size, d_in = input_shape[0], input_shape[-1]
        self.wq, self.wk = use_linear(units=d_in), use_linear(units=d_in)
        self.attention = L.Attention()
        self.memory = self.add_weight(
                shape=(batch_size, MEMORY_SIZE, d_in),
                initializer='random_normal', trainable=True)
        self.built = True

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x = tf.concat([x, self.memory], 1)
        attended = self.attention([self.wq(x), self.wk(x)])
        y, self.memory = tf.split(attended, [seq_len, MEMORY_SIZE], axis=1)
        return y

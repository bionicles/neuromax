import tensorflow as tf

from nature import Linear

L = tf.keras.layers

MEMORY_SIZE = 8


class AllAttention(L.Layer):
    def __init__(self):
        super(AllAttention, self).__init__()

    def build(self, input_shape):
        self.d_in = input_shape[-1]
        self.wq, self.wk = Linear(units=self.d_in), Linear(units=self.d_in)
        self.attention = L.Attention()
        self.memory = self.add_weight(
                shape=(1, MEMORY_SIZE, self.d_in),
                initializer='random_normal', trainable=True)
        self.built = True

    def call(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.d_in])
        seq_len = tf.shape(x)[1]
        x = tf.concat([x, self.memory], 1)
        attended = self.attention([self.wq(x), self.wk(x)])
        y, _ = tf.split(attended, [seq_len, MEMORY_SIZE], axis=1)
        return y

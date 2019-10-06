# https://gist.github.com/tam17aki/f8bebcc427f99a3432592e5ca0186cb8
# https://arxiv.org/pdf/1703.08864.pdf Ororbia et al 2017

import tensorflow as tf
import nature


L = tf.keras.layers
LAYER = nature.Layer
DROP_OPTIONS = [0., 0.5]


class Delta(L.Layer):

    def __init__(self, AI, units=None):
        super().__init__()
        self.p_drop = AI.pull("delta_p_drop", DROP_OPTIONS)
        self.ai = AI

    def build(self, shape):
        d_in = shape[-1]
        self.gate_bias = self.add_weight("gate_bias", [d_in], trainable=True)
        self.z_t_bias = self.add_weight("z_t_bias", [d_in], trainable=True)
        self.state = self.add_weight("state", shape, trainable=False)
        self.alpha = self.add_weight("alpha", [d_in], trainable=True)
        self.b1 = self.add_weight("b1", [d_in], trainable=True)
        self.b2 = self.add_weight("b2", [d_in], trainable=True)
        self.fc1 = LAYER(self.ai, units=d_in)
        self.fc2 = LAYER(self.ai, units=d_in)
        self.out = nature.Fn(self.ai)
        super().build(shape)

    @tf.function
    def call(self, x):
        # inner
        V_h = self.fc1(self.state)
        W_x = self.fc2(x)
        d1 = self.alpha * V_h * W_x
        d2 = self.b1 * V_h + self.b2 * W_x
        z_t = tf.nn.dropout(tf.nn.tanh(d1 + d2 + self.z_t_bias), self.p_drop)
        # outer
        gate = tf.nn.sigmoid(W_x + self.gate_bias)
        self.state.assign(self.out((1. - gate) * z_t + gate * self.state))
        return self.state

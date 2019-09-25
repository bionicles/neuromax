import tensorflow as tf
from tools import get_size
import nature

L = tf.keras.layers


class Delta(L.Layer):

    def __init__(self):
        super().__init__()

    def build(self, shape):
        size = get_size(shape)
        self.propose = nature.FC(units=size)
        self.update = nature.FC(units=size)
        self.state = tf.zeros(shape)
        self.built = True

    @tf.function
    def call(self, x):
        state_x = tf.concat([self.state, x], 1)
        proposal = self.propose(state_x)
        state_proposal = tf.concat([self.state, proposal], 1)
        self.state = self.update(state_proposal)
        return proposal

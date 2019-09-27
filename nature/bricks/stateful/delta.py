import tensorflow as tf
import nature
L = tf.keras.layers


class Delta(L.Layer):

    def __init__(self, units=None):
        super(Delta, self).__init__()

    def build(self, shape):
        self.batch = shape[0],
        self.state = self.add_weight("state", shape[1:], trainable=False)
        self.proposer = nature.Layer(units=shape[-1])
        self.updater = nature.Layer(units=shape[-1])
        self.fn = nature.Fn(key="tanh")
        self.concat = L.Concatenate(1)
        self.built = True

    @tf.function
    def delta_slice(self, x):
        proposal = self.proposer(self.concat([self.state, x]))
        proposal = self.fn(proposal)
        update = self.updater(self.concat([self.state, proposal]))
        update = self.fn(update)
        self.state.assign_add(update)
        return proposal

    @tf.function
    def call(self, inputs):
        return tf.map_fn(lambda x: self.delta_slice(x), inputs)

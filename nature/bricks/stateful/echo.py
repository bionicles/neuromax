# http://minds.jacobs-university.de/uploads/papers/PracticalESN.pdf
# http://www.scholarpedia.org/article/Echo_state_network
# https://arxiv.org/pdf/1906.03186.pdf
import tensorflow as tf
from tools import spectral_norm, make_id
import nature

L, B = tf.keras.layers, tf.keras.backend
UNITS = [128, 256, 512, 1024]
STDDEV, BACKOFF = 0.1, 0.005
LEAK_RATE = 0.04
P_DROP = 0.5


def dot_1D(x, weight, batch):
    return tf.tile(B.dot(tf.expand_dims(x, 0), weight), [batch, 1])


def get_weights(shape, dtype, stddev):
    weights = tf.nn.dropout(
        tf.random.truncated_normal(shape, stddev=stddev, dtype=dtype), P_DROP)
    norm = spectral_norm(weights)
    # log("NORM:", norm, color="yellow")
    return weights, norm


def get_init_echo_state(ai):
    desired_spectral_norm = ai.pull("spectral_norm", 0., 1.)

    def init_echo_state(shape, dtype=tf.float32):
        stddev = STDDEV
        weights, norm = get_weights(shape, dtype, stddev)
        while norm >= desired_spectral_norm:
            stddev = stddev - 0.01
            weights, norm = get_weights(shape, dtype, stddev)
        return weights
    return init_echo_state


class Echo(L.Layer):

    def __init__(self, AI, units=None):
        self.units = AI.pull(f"echo_units", UNITS)
        super().__init__(name=make_id(f"echo_{self.units}"))
        self.ai = AI

    def build(self, shape):
        self.batch = shape[0]
        self.y = self.add_weight("y", [self.batch, *self.ai.code_spec.shape])
        self.output_layer = nature.Resizer(self.ai, self.ai.code_spec.shape)
        self.adjacency = nature.Projector(
            self.units, init=get_init_echo_state(self.ai))
        self.state = self.add_weight("state", [self.batch, self.units])
        self.input_weights = nature.Projector(self.units)
        self.feedback = nature.Projector(self.units)
        self.flatten = L.Flatten()
        self.add = L.Add()
        super().build(shape)

    @tf.function
    def call(self, x):
        echo = self.adjacency(self.state)
        feedback = self.feedback(self.y)
        x = self.input_weights(x)
        echo = (1. - LEAK_RATE) * tf.nn.tanh(self.add([echo, feedback, x]))
        noise = tf.random.truncated_normal(tf.shape(echo), stddev=STDDEV)
        leak = LEAK_RATE * self.state
        new_state = self.add([noise, echo, leak])
        self.state.assign(new_state)
        self.y.assign(
            tf.nn.tanh(self.output_layer(tf.concat([new_state, x], axis=1))))
        return self.y

    # def compute_output_shape(self, shape):
    #     return (shape[0], self.units)

    # d_in = shape[1] * shape[2]
    # d_out = self.ai.code_spec.shape[-1]
    # self.input_weights = self.add_weight(
    #     "input_weights", [d_in, self.units],
    #     trainable=False, initializer=nature.Init(dist="truncated"))
    # self.state = self.add_weight(
    #     "state", [d_out, self.units],
    #     trainable=False, initializer=nature.Init())
    # self.state_shape = self.state.shape
    # self.adjacency = self.add_weight(
    #     "adjacency", [self.units, self.units],
    #     trainable=False, initializer=init_echo_state)
    # self.y = self.add_weight(
    #     "y", [self.batch, *self.ai.code_spec.shape], trainable=False)
    # self.feedback = self.add_weight(
    #     "feedback", [d_out, self.units],
    #     trainable=False, initializer=nature.Init())

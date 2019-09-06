import tensorflow as tf

from nature import Resizer, Coordinator, AllAttention, Norm

K = tf.keras
L = K.layers


class Predictor(L.Layer):
    """generates inputs and returns a surprise value"""

    def __init__(self, agent, in_spec):
        super(Predictor, self).__init__()
        self.in_spec = in_spec
        self.agent = agent

    def build(self, input_shape):
        self.resizer = Resizer(self.agent.code_spec.shape)
        rank = len(input_shape) - 1
        if rank > 1:
            self.coordinator = Coordinator()
        self.attention = AllAttention()
        self.noise_norm = Norm()
        self.x_norm = Norm()
        self.built = True

    def call(self, x):
        noise = tf.random.normal(x.shape)
        rank = len(x.shape) - 1
        if rank > 1:
            noise = self.coordinator(noise)
            x = self.coordinator(x)
        noise = self.noise_norm(noise)
        p = self.attention(noise)
        p = tf.reshape(p, x.shape)
        x = self.x_norm(x)
        surprise = x - p
        # self.add_loss(surprise)
        return self.resizer(surprise)

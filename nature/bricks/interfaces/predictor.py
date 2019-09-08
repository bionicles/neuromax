import tensorflow as tf

from nature import Resizer, ConcatCoords2D, ConcatCoords1D, AllAttention, Norm
from tools import concat_coords

K = tf.keras
L = K.layers

L1 = 1e-3
L2 = 1e-3

class Predictor(L.Layer):
    """generates inputs and returns a surprise value"""

    def __init__(self, agent, in_spec):
        super(Predictor, self).__init__()
        self.in_spec = in_spec
        self.agent = agent

    def build(self, input_shape):
        self.regularize = L.ActivityRegularization(l1=L1, l2=L2)
        self.resizer = Resizer(self.agent.code_spec.shape)
        if len(input_shape) is 4:
            coordinator = ConcatCoords2D
        elif len(input_shape) is 3:
            coordinator = ConcatCoords1D
        self.noise_coordinator = coordinator()
        self.x_coordinator = coordinator()
        self.attention = AllAttention()
        self.subtract = L.Subtract()
        self.noise_norm = Norm()
        self.x_norm = Norm()
        self.built = True

    def call(self, x):
        noise = tf.zeros_like(x)
        noise = self.noise_coordinator(noise)
        noise = self.noise_norm(noise)
        p = self.attention(noise)
        x = self.x_coordinator(x)
        p = tf.reshape(p, tf.shape(x))
        x = self.x_norm(x)
        surprise = self.subtract([x, p])
        surprise = self.regularize(surprise)
        return self.resizer(surprise)

    def compute_output_shape(self, input_shape):
        return self.agent.code_spec.shape

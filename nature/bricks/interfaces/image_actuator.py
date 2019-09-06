import tensorflow as tf

from nature import Resizer, DenseBlock, DConv2D

K = tf.keras
L, B = K.layers, K.backend

LAST_DECODER_FN = "tanh"


class ImageActuator(L.Layer):

    def __init__(self, agent):
        self.reshape = Resizer(agent, agent.image_spec.shape)
        self.dense_block = DenseBlock()
        self.deconv = DConv2D()

    def call(self, x):
        x = self.reshape(x)
        x = self.dense_block(x)
        return self.deconv(x)

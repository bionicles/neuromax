import tensorflow as tf

from nature import Resizer, DenseBlock, DConv2D, Brick

K = tf.keras
L, B = K.layers, K.backend

LAST_DECODER_FN = "tanh"


def ImageActuator(agent):
    reshape = Resizer(agent, agent.image_spec.shape)
    dense_block = DenseBlock()
    deconv = DConv2D()

    def call(self, x):
        x = reshape(x)
        x = dense_block(x)
        return deconv(x)
    return Brick(reshape, dense_block, deconv, call, agent)

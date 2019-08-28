import tensorflow as tf

from nature import use_resizer, use_dense_block, use_deconv_2D

K = tf.keras
L, B = K.layers, K.backend

LAST_DECODER_FN = "tanh"


def use_image_actuator(agent):
    reshape = use_resizer(agent.image_spec.shape)
    dense_block = use_dense_block()
    deconv = use_deconv_2D()

    def call(x):
        x = reshape(x)
        x = dense_block(x)
        return deconv(x)
    return call

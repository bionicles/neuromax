import tensorflow as tf

from nature import use_flatten_resize_reshape, use_deconv_2D

K = tf.keras
L, B = K.layers, K.backend

LAST_DECODER_FN = "tanh"


def use_image_actuator(agent, parts):
    # make layers
    parts.reshape = reshape = use_flatten_resize_reshape(parts.out_spec.shape)
    parts.dense_block = dense_block = agent.pull_brick("dense_block")
    deconv = use_deconv_2D(
        parts.out_spec.shape[-1], fn=LAST_DECODER_FN, padding="same")

    def call(self, x):
        x = reshape(x)
        x = dense_block(x)
        x = deconv(x)
        return x
    parts.call = call
    return parts

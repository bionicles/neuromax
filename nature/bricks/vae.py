import tensorflow as tf

from .residual_dense import get_dense_block_out
from .layers.conv import get_deconv_2D_out
from .layers.dense import get_dense_out

L, B = tf.keras.layers, tf.keras.backend

BLOCK_FN = get_dense_block_out
N_BLOCKS = 2

LAST_DECODER_FN = "tanh"


def sample_fn(args):
    code_mean, code_variance = args
    epsilon = tf.random.normal(tf.shape(code_variance))
    sample = code_mean + B.exp(0.5 * code_variance) * epsilon
    return sample


def get_image_encoder_out(agent, brick_id, input, out_spec, block_fn=BLOCK_FN):
    brick_id = f"{brick_id}_image_encoder"
    out = block_fn(agent, brick_id, input)
    for block_number in range(N_BLOCKS - 1):
        out = block_fn(agent, brick_id, out)
    flat_out = L.Flatten()(out)
    code_mean = get_dense_out(
        agent, brick_id + "_code_mean", units=out_spec.size)(flat_out)
    code_variance = get_dense_out(
        agent, brick_id + "_code_var", units=out_spec.size)(flat_out)
    sampled_tensor = L.Lambda(sample_fn, name='code')([
        code_mean, code_variance])
    code = L.Reshape(out_spec.shape)(sampled_tensor)
    return code


def get_image_decoder_out(agent, brick_id, input, out_spec):
    brick_id = f"{brick_id}_image_decoder"
    out = L.Flatten()(input)
    out = get_dense_out(agent, brick_id, out, units=out_spec.size)()
    out = L.Reshape(out_spec.shape)(out)
    for layer_number in range(N_BLOCKS):
        out = get_dense_block_out(  # dense skips for convexification
            agent, brick_id, out)
    out = get_deconv_2D_out(
        agent, f"{brick_id}_image_out_layer", out,
        filters=out_spec.shape[-1], fn=LAST_DECODER_FN, padding="same")
    return out

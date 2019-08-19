import tensorflow as tf

from nature.bricks.conv import get_deconv_2D_out, get_conv_2D_out
from nature.bricks.dense import get_dense

L, B = tf.keras.layers, tf.keras.backend

N_CONV_LAYERS = 4
N_DECONV_LAYERS = 2
LAST_DECODER_FN = "tanh"


def sample_fn(args):
    code_mean, code_variance = args
    epsilon = tf.random.normal(tf.shape(code_variance))
    sample = code_mean + B.exp(0.5 * code_variance) * epsilon
    return sample


def get_image_encoder_out(agent, brick_id, input, out_spec):
    brick_id = f"{brick_id}_image_encoder"
    out = get_conv_2D_out(agent, brick_id, input)
    for conv_layers in range(N_CONV_LAYERS - 1):
        out = get_conv_2D_out(agent, brick_id, out)
    flat_out = L.Flatten()(out)
    code_mean = get_dense(
        agent, brick_id + "_code_mean", units=out_spec.size)(flat_out)
    code_variance = get_dense(
        agent, brick_id + "_code_var", units=out_spec.size)(flat_out)
    sampled_tensor = L.Lambda(sample_fn, name='code')([
        code_mean, code_variance])
    code = L.Reshape(out_spec.shape)(sampled_tensor)
    return code


def get_image_decoder_out(agent, brick_id, input, out_spec):
    brick_id = f"{brick_id}_image_decoder"
    flattened_input = L.Flatten()(input)
    resized_code = get_dense(agent, brick_id, units=out_spec.size)(flattened_input)
    reshaped_code = L.Reshape(out_spec.shape)(resized_code)
    for layer_number in range(N_DECONV_LAYERS):
        reshaped_code = get_deconv_2D_out(
            agent, f"{brick_id}_deconv_{layer_number}", reshaped_code)
    image_out = get_deconv_2D_out(
        agent, f"{brick_id}_image_out_layer",
        reshaped_code, filters=out_spec.shape[-1], fn=LAST_DECODER_FN)
    return image_out

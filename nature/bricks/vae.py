import tensorflow as tf

from nature.bricks.conv_2D import get_deconv_2D_out, get_conv_2D_out
from nature.bricks.dense import get_dense_out

L, B = tf.keras.layers, tf.keras.backend

N_CONV_LAYERS = 2
N_DECONV_LAYERS = 2
LAST_DECODER_ACTIVATION = "sigmoid"


def sample_fn(args):
    code_mean, code_variance = args
    epsilon = tf.random.normal(tf.shape(code_variance))
    print("code_mean 1", code_mean)
    print("code_variance 1", code_variance)
    print("epsilon", epsilon)
    sample = code_mean + B.exp(0.5 * code_variance) * epsilon
    return sample


def get_image_encoder_output(agent, brick_id, input, out_spec):
    brick_id = f"{brick_id}_image_encoder"
    output = get_conv_2D_out(agent, brick_id, input)
    for conv_layers in range(N_CONV_LAYERS):
        output = get_conv_2D_out(agent, brick_id, output)
    flat_output = L.Flatten()(output)
    # x_output = get_dense_out(
    #     agent, brick_id, flat_output, units=out_spec.size)
    code_mean = get_dense_out(
        agent, brick_id + "_code_mean", flat_output, units=out_spec.size)
    code_variance = get_dense_out(
        agent, brick_id + "_code_var", flat_output, units=out_spec.size)
    print("code_mean 0", code_mean)
    print("code_variance 0", code_variance)
    sampled_tensor = L.Lambda(sample_fn, name='code')([
        code_mean, code_variance])
    code = L.Reshape(out_spec.shape)(sampled_tensor)
    return code


def get_image_decoder_output(agent, brick_id, input, out_spec):
    brick_id = f"{brick_id}_image_decoder"
    resized_code = get_dense_out(agent, brick_id, input, units=out_spec.size)
    reshaped_code = L.Reshape(out_spec.shape)(resized_code)
    for layer_number in range(N_DECONV_LAYERS):
        reshaped_code = get_deconv_2D_out(
            agent, f"{brick_id}_deconv_{layer_number}", reshaped_code)
    image_output = get_deconv_2D_out(
        agent, f"{brick_id}_image_output_layer",
        reshaped_code, filters=out_spec.shape[-1],
        activation=LAST_DECODER_ACTIVATION)
    return image_output

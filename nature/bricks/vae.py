import tensorflow as tf

from nature.bricks.preact_conv_2D import get_preact_deconv_2D_output
from nature.bricks.preact_conv_2D import get_preact_conv_2D_output
from nature.bricks.dense import get_dense_out

L, B = tf.keras.layers, tf.keras.backend

MIN_CONV_LAYERS, MAX_CONV_LAYERS = 4, 8
MIN_DECONV_LAYERS, MAX_DECONV_LAYERS = 4, 8
LAST_DECODER_ACTIVATION = "sigmoid"


def sample_fn(args):
    code_mean, code_variance = args
    epsilon = tf.random.normal(tf.shape(code_mean))
    sample = code_mean + B.exp(0.5 * code_variance) * epsilon
    return sample


def get_image_encoder_output(agent, brick_id, input, out_spec):
    brick_id = f"{brick_id}_image_encoder"
    n_conv_layers = agent.pull_numbers(
        f"{brick_id}_num_conv_layers",
        MIN_CONV_LAYERS, MAX_CONV_LAYERS)
    output = get_preact_conv_2D_output(agent, brick_id, input)
    for conv_layers in range(n_conv_layers):
        output = get_preact_conv_2D_output(agent, brick_id, output)
    flat_output = L.Flatten()(output)
    x_output = get_dense_out(
        agent, brick_id, flat_output, units=out_spec.size)
    code_mean = get_dense_out(
        agent, brick_id + "_code_mean", x_output, units=out_spec.size)
    code_variance = get_dense_out(
        agent, brick_id + "_code_var", x_output, units=out_spec.size)
    sampled_tensor = L.Lambda(
        sample_fn, output_shape=tf.shape(code_mean), name='code')(
        [code_mean, code_variance])
    code = L.Reshape(out_spec.shape)(sampled_tensor)
    return code


def get_image_decoder_output(agent, brick_id, input, out_spec):
    brick_id = f"{brick_id}_image_decoder"
    resized_code = get_dense_out(agent, brick_id, input, units=out_spec.size)
    reshaped_code = L.Reshape(out_spec.shape)(resized_code)
    n_deconv_layers = agent.pull_numbers(f"{brick_id}_num_deconv_layers",
                                         MIN_DECONV_LAYERS, MAX_DECONV_LAYERS)
    for layer_number in range(n_deconv_layers):
        reshaped_code = get_preact_deconv_2D_output(
            agent, f"{brick_id}_deconv_{layer_number}", reshaped_code)
    image_output = get_preact_deconv_2D_output(
        agent, f"{brick_id}_image_output_layer",
        reshaped_code, filters=out_spec.shape[-1],
        activation=LAST_DECODER_ACTIVATION)
    return image_output

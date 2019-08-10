import tensorflow_probability as tfp
from tools import get_size
import tensorflow as tf

from nature.bricks.preact_conv_2D import preact_conv_2D, preact_deconv_2D

L = tf.keras.layers
K = tf.keras.backend
tfd = tfp.distributions
tfpl = tfp.layers

MIN_FILTER_CONV, MAX_FILTER_CONV = 16, 64
MIN_FILTER_DECONV, MAX_FILTER_DECONV = 16, 64
MIN_CONV_LAYERS, MAX_CONV_LAYERS = 4, 8
MIN_DECONV_LAYERS, MAX_DECONV_LAYERS = 4, 8
MIN_X_LAYER_UNITS, MAX_X_LAYER_UNITS = 20, 1000
DENSE_ACTIVATION_OPTIONS = ["tanh"]
CONV_ACTIVATION_OPTIONS = ["relu", "sigmoid"]
DECONV_ACTIVATION_OPTIONS = ["relu", "sigmoid"]
LAST_DECODER_ACTIVATION = "sigmoid"


def sampling(args):
    z_mean, z_variance = args
    epsilon = tf.random.normal(tf.shape(z_mean))
    return z_mean + K.exp(0.5 * z_variance) * epsilon


def get_image_encoder_output(agent, brick_id, input, output_shape):
    dense_activation = agent.pull_choices(f"{brick_id}_dense_activation",
                                          DENSE_ACTIVATION_OPTIONS)
    conv_activation = agent.pull_choices(f"{brick_id}_conv_activation",
                                         CONV_ACTIVATION_OPTIONS)
    num_conv_layers = agent.pull_numbers(f"{brick_id}_num_conv_layers",
                                         MIN_CONV_LAYERS, MAX_CONV_LAYERS)
    filters_list = [
        agent.pull_numbers(f"{brick_id}_{layer_number}_filters",
                           MIN_FILTER_CONV, MAX_FILTER_CONV)
        for layer_number in range(num_conv_layers)]
    output = preact_conv_2D(input, filters=filters_list[0],
                            activation=conv_activation)(input)
    for filters in filters_list[1:]:
        output = preact_conv_2D(output, filters=filter,
                                activation=conv_activation)(output)
    flat_output = L.Flatten()(output)
    x_units = agent.pull_numbers(f"{brick_id}_x_layer_units",
                                 MIN_X_LAYER_UNITS, MAX_X_LAYER_UNITS)
    x_output = L.Dense(x_units, activation=dense_activation)(flat_output)
    code_mean = L.Dense(agent.code_spec.size, activation=dense_activation,
                        name='code_mean')(x_output)
    code_variance = L.Dense(agent.code_spec.size, activation=dense_activation,
                            name='code_variance')(x_output)
    code = L.Lambda(sampling, output_shape=output_shape,
                    name='code')([code_mean, code_variance])
    return code


def get_image_decoder_output(agent, brick_id, input, output_shape):
    deconv_activation = agent.pull_numbers(f"{brick_id}_conv_activation",
                                           DECONV_ACTIVATION_OPTIONS)
    dense_activation = agent.pull_choices(f"{brick_id}_dense_activation",
                                          DENSE_ACTIVATION_OPTIONS)
    resized_code = L.Dense(get_size(output_shape),
                           activation=dense_activation)(input)
    reshaped_code = L.Reshape(output_shape)(resized_code)
    num_deconv_layers = agent.pull_numbers(f"{brick_id}_num_deconv_layers",
                                           MIN_DECONV_LAYERS, MAX_DECONV_LAYERS)
    filters_list = [agent.pull_numbers(f"{brick_id}_{layer_number}_filters",
                                       MIN_FILTER_DECONV, MAX_FILTER_DECONV)
                    for layer_number in num_deconv_layers]
    for filters in filters_list:
        reshaped_code = preact_deconv_2D(reshaped_code, filters=filters,
                                         activation=deconv_activation)
    image_output = preact_deconv_2D(reshaped_code, filters=output_shape[-1],
                                    activation=LAST_DECODER_ACTIVATION)
    return image_output

import tensorflow as tf
import tensorflow_probability as tfp
from tools import get_size
from preact_conv import preact_conv2D, preact_deconv2D
L = tf.keras.layers
K = tf.keras.backend
tfd = tfp.distributions
tfpl = tfp.layers

MIN_FILTER_CONV, MAX_FILTER_CONV = 16, 64
MIN_FILTER_DECONV, MAX_FILTER_DECONV = 16, 64
MIN_CONV_LAYER, MAX_CONV_LAYER = 4, 8
MIN_DECONV_LAYER, MAX_DECONV_LAYER = 4, 8
MIN_X_LAYER_UNITS, MAX_X_LAYER_UNITS = 20, 1000
DENSE_ACTIVATION = ["tanh", "sigmoid"]
CONV_ACTIVATION = ["relu", "sigmoid"]
DECONV_ACTIVATION = ["relu", "sigmoid"]

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def get_image_encoder(agent, brick_id, input, output_shape):
    dense_activation = agent.pull_choices(f"{brick_id}_dense_activation", DENSE_ACTIVATION)
    conv_activation = agent.pull_choices(f"{brick_id}_conv_activation", CONV_ACTIVATION)
    code_size = agent.code_size
    num_conv_layers = agent.pull_numbers(f"{brick_id}_num_conv_layers", MIN_CONV_LAYER, MAX_CONV_LAYER)
    filters = [agent.pull_numbers(f"{brick_id}_{layer_number}_filters", MIN_FILTER_CONV, MAX_FILTER_CONV) for layer_number in num_conv_layers]
    conv_layer = preact_conv2D(input, filters=filters[0], activation=conv_activation)
    for i in range(1, num_conv_layers):
        conv_layer = preact_conv2D(conv_layer, filters=filters[i], activation=conv_activation)
    flatten_layer = L.Flatten()(conv_layer)
    x_layer = L.Dense(agent.pull_numbers(f"{brick_id}_x_layer_units", MIN_X_LAYER_UNITS, MAX_X_LAYER_UNITS), activation=dense_activation)(flatten_layer)
    z_mean = L.Dense(code_size, activation=dense_activation, name='z_mean')(x_layer)
    z_log_var = L.Dense(code_size, activation=dense_activation, name='z_log_var')(x_layer)
    code = L.Lambda(sampling, output_shape=output_shape, name='z')([z_mean, z_log_var])
    return code


def get_image_decoder_output(agent, brick_id, input, output_shape):
    deconv_activation = agent.pull_numbers(f"{brick_id}_conv_activation", DECONV_ACTIVATION)
    dense_activation = agent.pull_choices(f"{brick_id}_dense_activation", DENSE_ACTIVATION)
    reshape_dense = L.Dense(get_size(output_shape), activation=dense_activation)(input)
    reshape_latent = L.Reshape(output_shape)(reshape_dense)
    num_deconv_layers = agent.pull_numbers(f"{brick_id}_num_deconv_layers", MIN_DECONV_LAYER, MAX_DECONV_LAYER)
    filters = [agent.pull_numbers(f"{brick_id}_{layer_number}_filters", MIN_FILTER_DECONV, MAX_FILTER_DECONV) for layer_number in num_deconv_layers]
    for i in range(num_deconv_layers):
        reshape_latent = preact_deconv2D(reshape_latent, filters=filters[i], activation=deconv_activation)
    output = preact_deconv2D(reshape_latent, filters=output_shape[-1], activation="sigmoid")
    return output
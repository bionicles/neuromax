import tensorflow as tf
import tensorflow_probability as tfp
from preact_conv2D import preact_conv2D, preact_deconv2D
L = tf.keras.layers
K = tf.keras.backend
tfd = tfp.distributions
tfpl = tfp.layers

MIN_FILTERS, MAX_FILTERS = 16, 64
MIN_CONV_LAYERS, MAX_CONV_LAYERS = 4, 8
MIN_X_LAYER_UNITS, MAX_X_LAYER_UNITS = 20, 1000
DENSE_ACTIVATION = ["tanh", "sigmoid"]
CONV_ACTIVATION = ["relu", "sigmoid"]
DECONV_ACTIVATION = ["relu", "sigmoid"]


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def get_image_encoder(agent, input, output_shape):
    dense_activation = agent.pull_choices("dense_activation", DENSE_ACTIVATION)
    conv_activation = agent.pull_choices("conv_activation", CONV_ACTIVATION)
    code_size = agent.code_spec.size
    num_conv_layers = agent.pull_numbers("num_conv_layers", MIN_CONV_LAYERS, MAX_CONV_LAYERS)
    filters = [agent.pull_numbers("filters", MIN_FILTERS, MAX_FILTERS) for layer in num_conv_layers]
    conv_layer = preact_conv2D(input, filters=filters[0], activation=conv_activation)
    for i in range(1, num_conv_layers):
        filters *= 2
        conv_layer = preact_conv2D(conv_layer, filters=filters[i], activation=conv_activation)
    agent.parameters["last_conv_shape"] = K.int_shape(conv_layer)
    flatten_layer = L.Flatten()(conv_layer)
    x_layer = L.Dense(agent.pull_numbers("units", MIN_X_LAYER_UNITS, MAX_X_LAYER_UNITS), activation=dense_activation)(flatten_layer)
    z_mean = L.Dense(code_size, activation=dense_activation, name='z_mean')(x_layer)
    z_log_var = L.Dense(code_size, activation=dense_activation, name='z_log_var')(x_layer)
    return L.Lambda(sampling, output_shape=output_shape, name='z')([z_mean, z_log_var])


def get_image_decoder_output(agent, input, output_shape):
    deconv_activation = agent.pull_numbers("deconv_activation", DECONV_ACTIVATION)
    reshape_latent = L.Reshape(agent.parameters["last_conv_shape"][1:3])(input)
    filters = agent.parameters["filters"]
    filters.reverse()
    for i in range(agent.parameters["num_conv_layers"]):
        reshape_latent = preact_deconv2D(reshape_latent, filters=filters[i], activation=deconv_activation)
        filters //= 2
    output = preact_deconv2D(reshape_latent, filters=output_shape[-1], activation="sigmoid")
    return output

import tensorflow as tf
import tensorflow_probability as tfp
from preact_conv_2D import preact_conv_2D, preact_deconv_2D
L = tf.keras.layers
K = tf.keras.backend
tfd = tfp.distributions
tfpl = tfp.layers

MIN_FILTERS, MAX_FILTERS = 16, 64
MIN_CONV_LAYERS, MAX_CONV_LAYERS = 4, 8
MIN_X_LAYER_UNITS, MAX_X_LAYER_UNITS = 20, 1000
DENSE_ACTIVATION_OPTIONS = ["sigmoid"]
CONV_ACTIVATION_OPTIONS = ["relu", "sigmoid", "swish"]
DECONV_ACTIVATION_OPTIONS = ["relu", "sigmoid"]


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def get_image_encoder(agent, brick_id, input, output_shape):
    num_conv_layers = agent.pull_numbers(f"{brick_id}_num_conv_layers",
                                         MIN_CONV_LAYERS, MAX_CONV_LAYERS)
    filters = [agent.pull_numbers(f"{brick_id}_{layer_number}_filters",
                                  MIN_FILTERS, MAX_FILTERS)
               for layer_number in num_conv_layers]
    conv_activation = agent.pull_choices(
        f"{brick_id}_layer_0_conv_activation", CONV_ACTIVATION_OPTIONS)
    conv_layer = preact_conv_2D(input, filters=filters[0],
                                activation=conv_activation)
    for i in range(1, num_conv_layers):
        filters *= 2
        conv_activation = agent.pull_choices(
            f"{brick_id}_layer{i}_conv_activation", CONV_ACTIVATION_OPTIONS)
        conv_layer = preact_conv_2D(conv_layer, filters=filters[i],
                                    activation=conv_activation)
    agent.parameters[f"{brick_id}_last_conv_shape"] = K.int_shape(conv_layer)
    flatten_layer = L.Flatten()(conv_layer)
    activation = agent.pull_choices(f"{brick_id}_dense_activation",
                                          DENSE_ACTIVATION_OPTIONS)
    x_layer_units = agent.pull_numbers(f"{brick_id}_x_layer_units",
                                       MIN_X_LAYER_UNITS, MAX_X_LAYER_UNITS)
    x_layer = L.Dense(x_layer_units, activation=activation)(flatten_layer)
    code_size = agent.code_spec.size
    z_mean = L.Dense(code_size, activation=activation, name='z_mean')(x_layer)
    z_log_var = L.Dense(code_size, activation=activation, name='z_log_var')(x_layer)
    return L.Lambda(sampling, output_shape=output_shape, name='z')([z_mean, z_log_var])


def get_image_decoder_output(agent, input, output_shape):
    deconv_activation = agent.pull_numbers("deconv_activation", DECONV_ACTIVATION_OPTIONS)
    reshape_latent = L.Reshape(agent.parameters["last_conv_shape"][1:3])(input)
    filters = agent.parameters["filters"]
    filters.reverse()
    for i in range(agent.parameters["num_conv_layers"]):
        reshape_latent = preact_deconv_2D(reshape_latent, filters=filters[i], activation=deconv_activation)
        filters //= 2
    output = preact_deconv_2D(reshape_latent, filters=output_shape[-1], activation="sigmoid")
    return output

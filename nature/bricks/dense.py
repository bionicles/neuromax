import tensorflow_probability as tfp
import tensorflow as tf

from nature.bricks.activations import clean_activation

# tfd = tfp.distributions
tfpl = tfp.layers
K = tf.keras
B, L = K.backend, K.layers

FN_OPTIONS = ["tanh", "linear", "swish", "lisht"]
MIN_UNITS, MAX_UNITS = 8, 32
LAYER_OPTIONS = [tfpl.DenseReparameterization]


def get_dense_out(agent, brick_id, input, layer=None, units=None, fn=None,
                  fn_options=None):
    if layer is None:
        layer = agent.pull_choices(f"{brick_id}-layer_type", LAYER_OPTIONS)
    if units is None:
        units = agent.pull_numbers(f"{brick_id}-units", MIN_UNITS, MAX_UNITS)
    if fn is None:
        if fn_options is None:
            fn = agent.pull_choices(f"{brick_id}-fn", FN_OPTIONS)
        else:
            fn = agent.pull_choices(f"{brick_id}-fn", fn_options)
    fn = clean_activation(fn)
    return layer(units, activation=fn)(input)

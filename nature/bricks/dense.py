import tensorflow_probability as tfp
import tensorflow as tf

from nature.bricks.activations import clean_activation
from nature.bricks.noisedrop import NoiseDrop

# tfd = tfp.distributions
tfpl = tfp.layers
K = tf.keras
L = K.layers


LAYER_OPTIONS = [NoiseDrop, L.Dense, tfpl.DenseFlipout,
                 tfpl.DenseReparameterization]
FN_OPTIONS = ["tanh", "linear", "swish", "lisht", "sigmoid"]
MIN_STDDEV, MAX_STDDEV = 1e-4, 0.1
MIN_UNITS, MAX_UNITS = 32, 512


def get_dense_out(agent, brick_id, input, layer=None, units=None, fn=None,
                  fn_options=None):
    if layer is None:
        layer = agent.pull_choices(f"{brick_id}-layer_type", LAYER_OPTIONS)
    if units is None or units < MIN_UNITS or units > MAX_UNITS:
        units = agent.pull_numbers(f"{brick_id}-units", MIN_UNITS, MAX_UNITS)
    if fn is None:
        if fn_options is None:
            fn = agent.pull_choices(f"{brick_id}-fn", FN_OPTIONS)
        else:
            fn = agent.pull_choices(f"{brick_id}-fn", fn_options)
    fn = clean_activation(fn)
    if layer is NoiseDrop:
        stddev = agent.pull_numbers(f"{brick_id}-stddev",
                                    MIN_STDDEV, MAX_STDDEV)
        return NoiseDrop(units, activation=fn, stddev=stddev)(input)
    else:
        return layer(units, activation=fn)(input)
    raise Exception(f"get_layer failed on {brick_id} {layer} {units} {fn}")

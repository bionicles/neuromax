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


def get_layer(agent, brick_id, layer=None, units=None, fn=None):
    if layer not in LAYER_OPTIONS:
        layer = agent.pull_choices(f"{brick_id}-layer_type", LAYER_OPTIONS)
    if units is None or units < MIN_UNITS or units > MAX_UNITS:
        units = agent.pull_numbers(f"{brick_id}-units", MIN_UNITS, MAX_UNITS)
    if fn not in FN_OPTIONS:
        fn = agent.pull_choices(f"{brick_id}-fn", FN_OPTIONS)
    fn = clean_activation(fn)
    if layer is NoiseDrop:
        stddev = agent.pull_numbers(f"{brick_id}-stddev",
                                    MIN_STDDEV, MAX_STDDEV)
        return NoiseDrop(units, activation=fn, stddev=stddev)
    else:
        return layer(units, activation=fn)
    raise Exception(f"get_layer failed on {brick_id} {layer} {units} {fn}")

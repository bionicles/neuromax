import tensorflow as tf
import tensorflow_probability as tfp
from . import NoiseDrop

tfd = tfp.distributions
tfpl = tfp.layers

MIN_STDDEV, MAX_STDDEV = 1e-4, 0.1
MIN_UNITS, MAX_UNITS = 32, 512
LAYER_OPTIONS = ["dense", "noisedrop"]
FN_OPTIONS = ["tanh", "linear"]

K = tf.keras
L = K.layers


def get_layer(lkey, agent, ltype=None, units=None, fn=None, tfp_layer):
    if ltype not in LAYER_OPTIONS:
        ltype = agent.pull_choices(f"{lkey}-ltype", LAYER_OPTIONS)
    if units is None or units < MIN_UNITS or units > MAX_UNITS:
        units = agent.pull_numbers(f"{lkey}-units", MIN_UNITS, MAX_UNITS)
    if fn not in LAYER_OPTIONS:
        fn = agent.pull_numbers(f"{lkey}-fn", FN_OPTIONS)
    if ltype is "noisedrop":
        stddev = agent.pull_numbers(f"{lkey}-stddev", MIN_STDDEV, MAX_STDDEV)
        return NoiseDrop(units, activation=fn, stddev=stddev)
    if ltype is "dense":
        if tfp_layer:
            return tfpl.DenseFlipout(units, activation=fn)
        else:
            return L.Dense(units, activation=fn)
    raise Exception(f"get_layer failed on {lkey} {ltype} {units} {fn}")

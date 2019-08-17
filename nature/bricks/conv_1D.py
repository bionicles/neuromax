import tensorflow_probability as tfp
import tensorflow as tf

MIN_FILTERS, MAX_FILTERS = 4, 16
FN = "tanh"

L = tf.keras.layers
tfpl = tfp.layers


def get_conv_1D(agent, brick_id, d_out=None, fn=FN):
    if d_out is None:
        d_out = agent.pull_numbers(f"{id}_filters", MIN_FILTERS, MAX_FILTERS)
    brick = tfpl.Convolution1DReparameterization(d_out, 1, activation=fn)
    return brick

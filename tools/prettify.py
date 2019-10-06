import tensorflow as tf

DECIMALS = 2


def prettify(x):
    x = tf.math.reduce_mean(x)
    multiplier = tf.constant(10**DECIMALS, dtype=x.dtype)
    x = tf.round(x * multiplier) / multiplier
    x = tf.strings.as_string(x, precision=DECIMALS)
    return x

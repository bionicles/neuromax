import tensorflow as tf


@tf.function
def normalize(a):
    return (a - tf.math.reduce_mean(a)) / tf.math.reduce_mean(a)

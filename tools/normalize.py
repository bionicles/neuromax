import tensorflow as tf


def normalize(a):
    return (a - tf.math.reduce_mean(a)) / tf.math.reduce_max(a)

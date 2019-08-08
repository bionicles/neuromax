import tensorflow as tf


def sum_entropy(actions):
    return tf.math.reduce_sum([action.entropy() for action in actions])

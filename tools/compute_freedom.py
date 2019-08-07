import tensorflow as tf


def compute_freedom(actions):
    return tf.math.reduce_sum([action.entropy() for action in actions])

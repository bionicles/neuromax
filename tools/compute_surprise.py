import tensorflow as tf


def compute_surprise(beliefs, truths):
    """Return sum of negative log probabilities of truths given beliefs"""
    return tf.math.reduce_sum([-1 * belief.log_prob(truth)
                               for belief, truth in zip(beliefs, truths)])

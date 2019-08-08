from tensorflow_probability import distributions as tfd
import tensorflow as tf


def get_prior(size, prior_type="normal"):
    """
    Args:
        size: int, the event size
        prior_type: string, the type of prior
    Returns: a prior distribution
    """

    if prior_type is "normal":
        prior = tfd.Independent(tfd.Normal(size))
    else:
        ones = tf.ones(size)
        prior = tfd.Independent(
            tfd.Uniform(low=(-1 * ones), high=ones),
            reinterpreted_batch_ndims=1)
    return prior

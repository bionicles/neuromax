from tensorflow_probability import distributions as tfd
import tensorflow as tf

from .get_size import get_size


def get_prior(desired_shape, prior_type="normal"):
    """
    Args:
        size: int, the event size
        prior_type: string, the type of prior
    Returns: a prior distribution
    """
    if prior_type is "normal":
        shapes = tfd.Normal.param_shapes(desired_shape)
        loc, scale = tf.zeros(shapes["loc"]), tf.ones(shapes["scale"])
        return tfd.Normal(loc, scale), shapes
    elif "categorical" in prior_type:
        n = get_size(desired_shape)
        probs = [1 / n for _ in range(n)]
        if "one_hot" in prior_type:
            return tfd.OneHotCategorical(probs=probs), n
        else:
            return tfd.Categorical(probs=probs), n

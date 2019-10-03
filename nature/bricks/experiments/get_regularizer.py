from tensorflow_probability import layers as tfpl

from . import get_prior


def get_regularizer(size, prior_type="normal"):
    prior = get_prior(size, prior_type)
    return tfpl.KLDivergenceRegularizer(prior)

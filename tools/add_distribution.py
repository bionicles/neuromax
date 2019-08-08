import tensorflow_probability as tfp

from . import get_prior

tfd = tfp.distributions
tfpl = tfp.layers


def add_distribution(model, size):
    prior = get_prior(size)
    model.add(tfpl.MultivariateNormalTriL(
        size, activity_regularizer=tfpl.KLDivergenceRegularizer(prior)))
    return model

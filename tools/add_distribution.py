import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers


def add_distribution(model, size):
    ones = tf.ones(size)
    prior = tfd.Independent(
        tfd.Uniform(low=(-1 * ones), high=ones),
        reinterpreted_batch_ndims=1)
    model.add(tfpl.MultivariateNormalTriL(
        size, activity_regularizer=tfpl.KLDivergenceRegularizer(prior)))
    return model

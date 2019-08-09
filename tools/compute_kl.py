import tensorflow_probability as tfp


def compute_kl(a, b):
    return tfp.distributions.kl_divergence(a, b)

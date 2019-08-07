from tensorflow_probability.distributions import kl_divergence


def compute_kl(a, b):
    return kl_divergence(a, b)

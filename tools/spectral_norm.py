import tensorflow as tf


def spectral_norm(x):
    return tf.linalg.svd(x, compute_uv=False)[..., 0]

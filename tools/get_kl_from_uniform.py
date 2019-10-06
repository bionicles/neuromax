import tensorflow as tf

from .get_uniform import get_uniform

K = tf.keras

@tf.function
def get_kl_from_uniform(x):
    uniform = get_uniform(x)
    return tf.losses.KLD(uniform, x)

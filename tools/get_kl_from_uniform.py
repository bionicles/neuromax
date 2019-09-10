import tensorflow as tf

K = tf.keras

@tf.function
def get_kl_from_uniform(x):
    ones = tf.ones_like(x, dtype=tf.float32)
    uniform = ones / tf.math.reduce_sum(ones)
    return tf.losses.KLD(uniform, x)

import tensorflow as tf

DIST = tf.random.uniform
MIN, MAX = -2., 2.

@tf.function
def noise_like(x):
    return DIST(tf.shape(x), minval=MIN, maxval=MAX)

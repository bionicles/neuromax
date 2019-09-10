import tensorflow as tf


@tf.function
def tile_for_batch(batch, x):
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, [batch, 1, 1, 1])
    return x

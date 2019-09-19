import tensorflow as tf

from .tile_for_batch import tile_for_batch

def get_uniform(x, batch=None):
    shape = x
    if tf.is_tensor(x):
        shape = tf.shape(x)
    ones = tf.ones(shape, dtype=tf.float32)
    uniform = ones / tf.math.reduce_sum(ones)
    if batch:
        uniform = tile_for_batch(batch, uniform)
    return uniform

import tensorflow as tf
from .normalize import normalize


@tf.function
def concat_1D_coords(tensor, should_normalize=True):
    """ (B, W, C) ---> (B, W, C+1) with i coordinates"""
    width = tf.shape(tensor)[1]
    coords = tf.range(width)
    coords = tf.cast(coords, tf.float32)
    coords = tf.expand_dims(coords, 0)
    coords = tf.expand_dims(coords, -1)
    if should_normalize:
        coords = normalize(coords)
    return tf.concat([tensor, coords], -1)

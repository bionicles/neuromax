import tensorflow as tf
from . import normalize


@tf.function
def concat_1D_coords(tensor, should_normalize=False):
    """ (W, C) ---> (W, C+1) with i coordinates"""
    width = tf.shape(tensor)[0]
    coords = width
    if should_normalize:
        coords = normalize(coords)
    return tf.concat([tensor, coords], -1)

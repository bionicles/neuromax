import tensorflow as tf


@tf.function
def concat_1D_coords(tensor, normalize=False):
    """ (W, C) ---> (W, C+1) with i coordinates"""
    width = tf.shape(tensor)[0]
    coords = width
    if normalize:
        coords = coords / width
    return tf.concat([tensor, coords], -1)

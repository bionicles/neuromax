import tensorflow as tf

from . import normalize


def get_2D_coords(a, b, should_normalize=False):
    a = tf.range(a)
    a = tf.cast(a, tf.float32)
    b = tf.range(b)
    if should_normalize:
        a = normalize(a)
        b = normalize(b)
    b = tf.cast(b, tf.float32)
    return tf.reshape(tf.stack(tf.meshgrid(a, b, indexing='ij'), axis=-1), (-1, 2))


def concat_2D_coords(tensor):
    """ (H, W, C) ---> (H, W, C+2) with i,j coordinates"""
    in_shape = tensor.shape
    coords = get_2D_coords(*in_shape, should_normalize=True)
    return tf.concat([tensor, coords], -1)

import tensorflow as tf

from . import normalize


@tf.function
def get_cartesian_product(a, b, should_normalize=False):
    a = tf.range(a)
    a = tf.cast(a, tf.float32)
    b = tf.range(b)
    if should_normalize:
        a = normalize(a)
        b = normalize(b)
    b = tf.cast(b, tf.float32)
    return tf.reshape(tf.stack(tf.meshgrid(a, b, indexing='ij'), axis=-1), (-1, 2))


@tf.function
def concat_2D_coords(tensor):
    """ (H, W, C) ---> (H, W, C+2) with i,j coordinates"""
    in_shape = tf.shape(tensor)
    coords = get_cartesian_product(*in_shape, normalize=True)
    return tf.concat([tensor, coords], -1)

import tensorflow as tf

from .get_image_hw import get_image_hw
from .normalize import normalize


def get_2D_coords(a, b, should_normalize=False):
    a = tf.range(a)
    a = tf.cast(a, tf.float32)
    b = tf.range(b)
    if should_normalize:
        a = normalize(a)
        b = normalize(b)
    b = tf.cast(b, tf.float32)
    return tf.stack(tf.meshgrid(a, b, indexing='ij'), axis=-1)


def concat_2D_coords(tensor):
    """
    (B, H, W, C) ---> (B, H, W, C+2) with i,j coordinates
    (H, W, C) ---> (H, W, C+2) with i,j coordinates
    """
    h, w = get_image_hw(tensor)
    coords = get_2D_coords(h, w, should_normalize=True)
    if len(tensor.shape) is 4:
        coords = tf.expand_dims(coords, 0)
    return tf.concat([tensor, coords], -1)

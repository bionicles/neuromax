import tensorflow as tf

from .get_hw import get_hw
from .normalize import normalize
from .get_size import get_size
from .log import log


def concat_coords(tensor):
    log("concat coords to tensor", tensor)
    if len(tensor.shape) in [2, 3]:  # B, W, C
        if len(tensor.shape) is 2:
            tensor = tf.expand_dims(tensor, -1)
            if get_size(tensor.shape) is 1:
                return tensor
        return concat_1D_coords(tensor)
    elif len(tensor.shape) is 4:  # B, H, W, C
        return concat_2D_coords(tensor)


def concat_1D_coords(tensor):
    """ (B, W, C) ---> (B, W, C+1) with i coordinates"""
    log("concat_1D_coords", color="white")
    width = tf.shape(tensor)[1]
    coords = tf.range(width, dtype=tf.float32)
    coords = tf.expand_dims(coords, 0)
    coords = tf.expand_dims(coords, -1)
    coords = normalize(coords)
    if len(tensor.shape) < len(coords.shape):
        tensor = tf.expand_dims(tensor, -1)
    concatenated = tf.concat([tensor, coords], -1)
    log("concatenated shape", concatenated.shape, color="white")
    return concatenated


def get_2D_coords(a, b):
    a = tf.range(a, dtype=tf.float32)
    b = tf.range(b, dtype=tf.float32)
    a = normalize(a)
    b = normalize(b)
    coords = tf.stack(tf.meshgrid(a, b, indexing='ij'), axis=-1)
    return coords


def concat_2D_coords(tensor):
    """
    (B, H, W, C) ---> (B, H, W, C+2) with i,j coordinates
    (H, W, C) ---> (H, W, C+2) with i,j coordinates
    """
    log("concat_1D_coords", color="white")
    h, w = get_hw(tensor)
    coords = get_2D_coords(h, w, should_normalize=True)
    if len(tensor.shape) is 4:
        coords = tf.expand_dims(coords, 0)
    concatenated = tf.concat([tensor, coords], -1)
    log("concatenated shape", concatenated.shape, color="white")
    return concatenated

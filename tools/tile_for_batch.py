import tensorflow as tf
from tools import log


@tf.function
def tile_for_batch(batch, x):
    multiples = [batch] + [1 for _ in x.shape]
    x = tf.expand_dims(x, 0)
    x = tf.tile(x, multiples)
    return x

import tensorflow as tf


def tile_to_batch_size(batch_size, x):
    x = tf.expand_dims(x, 0)
    multiples = [1 for _ in range(len(x.shape))]
    multiples[0] = batch_size
    return tf.tile(x, multiples)

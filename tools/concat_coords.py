import tensorflow as tf
import numpy as np

@tf.function
def concat_coords(x):
    coords = tf.numpy_function(get_coords, [x], tf.float32)
    return tf.concat([x, coords], -1)

def get_coords(x):
    shape = tf.shape(x)[1], tf.shape(x)[2]
    coords = np.indices(shape)
    coords = np.transpose(coords)
    return np.expand_dims(coords, 0)

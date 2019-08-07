import tensorflow as tf
from . import get_size
L = tf.keras.layers


def flatten_resize_reshape(output, out_shape):
    output = L.Flatten()(output)
    output = L.Dense(get_size(out_shape))(output)
    output = L.Reshape(out_shape)(output)
    assert output.shape == out_shape
    return output

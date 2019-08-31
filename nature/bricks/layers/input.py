import tensorflow as tf


def Input(tensor_or_shape, batch_size=1, drop_batch_dim=False):
    if tf.is_tensor(tensor_or_shape):
        shape = tensor_or_shape.shape
    else:
        shape = tensor_or_shape
    if drop_batch_dim:
        shape = shape[1:]
    return tf.keras.Input(shape, batch_size=batch_size, dtype=tf.float32)

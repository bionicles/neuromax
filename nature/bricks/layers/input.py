import tensorflow as tf


def use_input(tensor_or_shape, batch_size=1):
    if tf.is_tensor(tensor_or_shape):
        shape = tensor_or_shape.shape
    else:
        shape = tensor_or_shape
    return tf.keras.Input(shape[1:], batch_size=batch_size)

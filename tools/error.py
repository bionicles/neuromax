import tensorflow as tf


@tf.function(experimental_relax_shapes=True)
def error(true, pred):
    return tf.math.abs(tf.math.subtract(true, pred))

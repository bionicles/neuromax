import tensorflow as tf

from .get_value import get_value

def add_up(tensors):
    values = [get_value(tf.reduce_sum(t)) for t in tensors]
    return sum(values)

@tf.function
def add_up_graph(tensors):
    tensors = tf.map_fn(
        tf.math.reduce_sum, tensors, back_prop=False, infer_shape=False)
    return tf.math.add_n(tensors)

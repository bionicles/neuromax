import tensorflow as tf

from .get_value import get_value


def add_up(tensors):
    values = [get_value(tf.reduce_sum(t)) for t in tensors]
    return sum(values)

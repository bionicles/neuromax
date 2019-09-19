import tensorflow as tf
from nature import L1L2, Init

K, L = tf.keras, tf.keras.layers
REGULARIZER = L1L2
INITIALIZER = Init

ONE_D_LAYER_CLASS = L.SeparableConv1D
ONE_D_KERNEL_SIZE = 3
ONE_D_PADDING = "same"
ONE_D_FILTERS = 16


def Conv1D(
        units=ONE_D_FILTERS,
        layer_class=ONE_D_LAYER_CLASS, kernel_size=ONE_D_KERNEL_SIZE,
        padding=ONE_D_PADDING,
        regularizer=REGULARIZER, initializer=INITIALIZER):
    return layer_class(
        filters=units, kernel_size=kernel_size, padding=padding,
        kernel_regularizer=regularizer(),
        activity_regularizer=regularizer(),
        bias_regularizer=regularizer(),
        kernel_initializer=initializer(),
        bias_initializer=initializer())

import tensorflow as tf

from nature import L1L2, Init

K, L = tf.keras, tf.keras.layers

REGULARIZER = L1L2
INITIALIZER = Init

DECONV_TWO_D_LAYER_CLASS = L.Conv2DTranspose
DECONV_TWO_D_KERNEL_SIZE = 4
DECONV_TWO_D_PADDING = "same"
DECONV_TWO_D_FILTERS = 4


def DConv2D(
        units=DECONV_TWO_D_FILTERS,
        layer_class=DECONV_TWO_D_LAYER_CLASS,
        kernel_size=DECONV_TWO_D_KERNEL_SIZE,
        padding=DECONV_TWO_D_PADDING,
        regularizer=REGULARIZER, initializer=INITIALIZER):
    return layer_class(
        filters=units, kernel_size=kernel_size, padding=padding,
        kernel_regularizer=regularizer(),
        # activity_regularizer=regularizer(),
        bias_regularizer=regularizer(),
        kernel_initializer=initializer(),
        bias_initializer=initializer())

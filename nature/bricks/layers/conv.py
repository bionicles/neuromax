import tensorflow as tf

from nature import get_l1_l2, get_chaos

K = tf.keras
L = K.layers

ONE_D_LAYER_CLASS = L.SeparableConv1D
ONE_D_KERNEL_SIZE = 1
ONE_D_PADDING = None
ONE_D_FILTERS = 16

CONV_TWO_D_LAYER_CLASS = L.SeparableConv2D
CONV_TWO_D_KERNEL_SIZE = 4
CONV_TWO_D_PADDING = "valid"
CONV_TWO_D_FILTERS = 4

DECONV_TWO_D_LAYER_CLASS = L.Conv2DTranspose
DECONV_TWO_D_KERNEL_SIZE = 4
DECONV_TWO_D_PADDING = "same"
DECONV_TWO_D_FILTERS = 4


def use_conv_1D(
        layer_class=ONE_D_LAYER_CLASS,
        filters=ONE_D_FILTERS, kernel_size=ONE_D_KERNEL_SIZE,
        padding=ONE_D_PADDING):
    return layer_class(
        filters=filters, kernel_size=kernel_size, padding=padding,
        kernel_regularizer=get_l1_l2(),
        activity_regularizer=get_l1_l2(),
        bias_regularizer=get_l1_l2(),
        kernel_initializer=get_chaos(),
        bias_initializer=get_chaos(bias=True))


def use_conv_2D(
        layer_class=CONV_TWO_D_LAYER_CLASS,
        kernel_size=CONV_TWO_D_KERNEL_SIZE, filters=CONV_TWO_D_FILTERS,
        padding=CONV_TWO_D_PADDING):
    return layer_class(
        filters=filters, kernel_size=kernel_size, padding=padding,
        kernel_regularizer=get_l1_l2(),
        activity_regularizer=get_l1_l2(),
        bias_regularizer=get_l1_l2(),
        kernel_initializer=get_chaos(),
        bias_initializer=get_chaos(bias=True))


def use_deconv_2D(
        layer_class=DECONV_TWO_D_LAYER_CLASS,
        kernel_size=DECONV_TWO_D_KERNEL_SIZE, filters=DECONV_TWO_D_FILTERS,
        padding=DECONV_TWO_D_PADDING):
    return layer_class(
        filters=filters, kernel_size=kernel_size, padding=padding,
        kernel_regularizer=get_l1_l2(),
        activity_regularizer=get_l1_l2(),
        bias_regularizer=get_l1_l2(),
        kernel_initializer=get_chaos(),
        bias_initializer=get_chaos(bias=True))

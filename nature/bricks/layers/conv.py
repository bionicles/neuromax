import tensorflow as tf

from .helpers.regularize import get_l1_l2
from .helpers.chaos import get_chaos

from tools.get_brick import get_brick

K = tf.keras
L = K.layers

FN = None

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


def get_conv_1D_out(
        agent, id, out, layer_class=ONE_D_LAYER_CLASS,
        filters=ONE_D_FILTERS, kernel_size=ONE_D_KERNEL_SIZE,
        padding=ONE_D_PADDING, return_brick=False):

    layer = layer_class(
        filters=filters, kernel_size=kernel_size, padding=padding,
        kernel_regularizer=get_l1_l2(),
        activity_regularizer=get_l1_l2(),
        bias_regularizer=get_l1_l2(),
        kernel_initializer=get_chaos(),
        bias_initializer=get_chaos(bias=True)
    )

    def convolve(out):
        return layer(out)
    return get_brick(convolve, out, return_brick)


def get_conv_2D_out(
        agent, brick_id, out, layer_class=CONV_TWO_D_LAYER_CLASS,
        kernel_size=CONV_TWO_D_KERNEL_SIZE, filters=CONV_TWO_D_FILTERS,
        padding=CONV_TWO_D_PADDING, return_brick=False):

    layer = layer_class(
        filters=filters, kernel_size=kernel_size, padding=padding,
        kernel_regularizer=get_l1_l2(),
        activity_regularizer=get_l1_l2(),
        bias_regularizer=get_l1_l2(),
        kernel_initializer=get_chaos(),
        bias_initializer=get_chaos(bias=True)
    )

    def convolve(out):
        return layer(out)
    return get_brick(convolve, out, return_brick)


def get_deconv_2D_out(
        agent, brick_id, out, layer_class=DECONV_TWO_D_LAYER_CLASS,
        kernel_size=DECONV_TWO_D_KERNEL_SIZE, filters=DECONV_TWO_D_FILTERS,
        padding=DECONV_TWO_D_PADDING, fn=FN, return_brick=False):

    layer = layer_class(
        filters=filters, kernel_size=kernel_size, padding=padding,
        kernel_regularizer=get_l1_l2(),
        activity_regularizer=get_l1_l2(),
        bias_regularizer=get_l1_l2(),
        kernel_initializer=get_chaos(),
        bias_initializer=get_chaos(bias=True)
    )

    def convolve(out):
        return layer(out)
    return get_brick(convolve, out, return_brick)

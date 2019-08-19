from tensorflow_addons.layers import InstanceNormalization
import tensorflow_probability as tfp
import tensorflow as tf

from .chaos import EdgeOfChaos
from .activations import swish

K = tf.keras
L = K.layers
tfpl = tfp.layers

TFP_LAYER = tfpl.Convolution2DReparameterization
ONE_D_KERNEL_SIZE = 1
TWO_D_KERNEL_SIZE = 4
NORM = "instance"
PADDING = "valid"
FILTERS = 32
FN = swish

KERNEL_STDDEV = 2.952
BIAS_STDDEV = 0.04
L1, L2 = 0.001, 0.001


def get_conv_1D(agent, brick_id, d_out, k=ONE_D_KERNEL_SIZE, fn=FN):
    if agent.probabilistic:
        layer = tfpl.Convolution1DReparameterization
    else:
        layer = L.SeparableConv1D
    brick = layer(d_out, k, activation=fn,
                  kernel_regularizer=K.regularizers.L1L2(l1=L1, l2=L2),
                  kernel_initializer=EdgeOfChaos(True, "swish"),
                  bias_initializer=EdgeOfChaos(False, "swish"))
    return brick


def get_conv_2D_out(agent, brick_id, input, fn=FN,
                    k=TWO_D_KERNEL_SIZE, filters=FILTERS, norm=NORM,
                    tfp_layer=TFP_LAYER, padding=PADDING):
    if norm is "instance":
        output = InstanceNormalization()(input)
    elif norm is "batch":
        output = L.BatchNormalization()(input)
    else:
        output = input
    output = L.Activation(fn)(output)
    layer = tfp_layer if agent.probabilistic else L.SeparableConv2D
    return layer(filters, kernel_size=k, padding=padding,
                 kernel_regularizer=K.regularizers.L1L2(l1=L1, l2=L2),
                 kernel_initializer=EdgeOfChaos(True, "swish"),
                 bias_initializer=EdgeOfChaos(False, "swish"))(output)


def get_deconv_2D_out(agent, brick_id, input, fn=FN,
                      k=TWO_D_KERNEL_SIZE, filters=FILTERS, norm=NORM,
                      padding=PADDING):
    if norm is "instance":
        output = InstanceNormalization()(input)
    elif norm is "batch":
        output = L.BatchNormalization()(input)
    else:
        output = input
    output = L.Activation(fn)(output)
    return L.Conv2DTranspose(
        filters, kernel_size=k, padding=padding,
        kernel_regularizer=K.regularizers.L1L2(l1=L1, l2=L2),
        kernel_initializer=EdgeOfChaos(True, "swish"),
        bias_initializer=EdgeOfChaos(False, "swish"))(output)

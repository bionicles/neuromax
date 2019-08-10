# from tensorlayer.layers import DeformableConv2d
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import tensorflow as tf

InstanceNormalization = tfa.layers.InstanceNormalization
L = tf.keras.layers
tfpl = tfp.layers

TFP_LAYER = None
ACTIVATION = "relu"
NORM = "instance"
PADDING = "same"
KERNEL_SIZE = 3
FILTERS = 64


def get_preact_conv_2D_output(agent, brick_id, input, activation=ACTIVATION,
                              k=KERNEL_SIZE, filters=FILTERS, norm=NORM,
                              tfp_layer=TFP_LAYER, padding=PADDING):
    if norm is "instance":
        output = InstanceNormalization()(input)
    else:
        output = L.BatchNormalization()(input)
    output = L.Activation(activation)(output)
    if tfp_layer is None:
        return L.Conv2D(filters, kernel_size=(k, k), padding=padding)(output)
    else:
        return tfp_layer(filters, kernel_size=(k, k), padding=padding)(output)


def get_preact_deconv_2D_output(agent, brick_id, input, activation=ACTIVATION,
                                k=KERNEL_SIZE, filters=FILTERS, norm=NORM,
                                tfp_layer=TFP_LAYER, padding=PADDING):
    if norm is "instance":
        output = InstanceNormalization()(input)
    else:
        output = L.BatchNormalization()(input)
    return L.Conv2DTranspose(filters, kernel_size=(k, k), padding=padding,
                             activation=activation)(output)

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
PADDING = "valid"
KERNEL_SIZE = 3
FILTERS = 16

# TODO: add "rank" so we can reuse this for 1D conv
def get_conv_2D_out(agent, brick_id, input, activation=ACTIVATION,
                              k=KERNEL_SIZE, filters=FILTERS, norm=NORM,
                              tfp_layer=TFP_LAYER, padding=PADDING):
    if norm is "instance":
        output = InstanceNormalization()(input)
    else:
        output = L.BatchNormalization()(input)
    output = L.Activation(activation)(output)
    if tfp_layer is None:
        return L.Conv2D(filters, kernel_size=k, padding=padding)(output)
    else:
        return tfp_layer(filters, kernel_size=k, padding=padding)(output)


def get_deconv_2D_out(agent, brick_id, input, activation=ACTIVATION,
                                k=KERNEL_SIZE, filters=FILTERS, norm=NORM,
                                tfp_layer=TFP_LAYER, padding=PADDING):
    if norm is "instance":
        output = InstanceNormalization()(input)
    else:
        output = L.BatchNormalization()(input)
    return L.Conv2DTranspose(filters, kernel_size=k, padding=padding,
                             activation=activation)(output)

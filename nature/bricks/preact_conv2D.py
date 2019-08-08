from tensorflow_addons import InstanceNormalization
import tensorflow as tf
import tensorflow_probability as tfp
tfpl = tfp.layers
K = tf.keras
L = K.layers

TFP_LAYER = tfpl.Conv2DFlipout
ACTIVATION = "relu"
NORM = "instance"
PADDING = "same"
KERNEL_SIZE = 3
FILTERS = 64


def preact_conv2D(inputs, activation=ACTIVATION, k=KERNEL_SIZE,
                  filters=FILTERS, norm=NORM, tfp_layer=TFP_LAYER,
                  padding=PADDING):
    if norm is "instance":
        outputs = InstanceNormalization()(inputs)
    else:
        outputs = L.BatchNormalization()(inputs)
    outputs = L.Activation(activation)(outputs)
    if tfp_layer is not None:
        return tfp_layer(filters, kernel_size=(k, k), padding=padding)(outputs)
    else:
        return L.Conv2D(filters, kernel_size=(k, k), padding=padding)(outputs)


def preact_deconv2D(inputs, activation=ACTIVATION, k=KERNEL_SIZE,
                    filters=FILTERS, norm=NORM, tfp_layer=TFP_LAYER,
                    padding=PADDING):
    if norm is "instance":
        outputs = InstanceNormalization()(inputs)
    else:
        outputs = L.BatchNormalization()(inputs)
    return L.Conv2DTranspose(filters, kernel_size=(k, k), padding=padding,
                             activation=activation)(outputs)

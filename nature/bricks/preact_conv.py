from tensorflow_addons import InstanceNormalization
import tensorflow as tf
import tensorflow_probability as tfp
tfpl = tfp.layers
K = tf.keras
L = K.layers


def preact_conv2D(inputs, k=3, filters=64, norm="instance", tfp_layer=False, activation):
    if norm is "instance":
        outputs = InstanceNormalization()(inputs)
    else:
        outputs = L.BatchNormalization()(inputs)
    outputs = L.Activation(activation)(outputs)
    if tfp_layer:
        return tfpl.Convolution2DFlipout(filters, kernel_size=(k, k), padding='same')(outputs)
    else:
        return L.Conv2D(filters, kernel_size=(k, k), padding='same')(outputs)


def preact_deconv2D(inputs, k=3, filters=64, norm="instance", activation="relu"):
    if norm is "instance":
        outputs = InstanceNormalization()(inputs)
    else:
        outputs = L.BatchNormalization()(inputs)
    return L.Conv2DTranspose(filters, kernel_size=(k, k), padding='same', activation=activation)(outputs)
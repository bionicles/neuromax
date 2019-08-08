from tensorflow_addons import InstanceNormalization
import tensorflow as tf

K = tf.keras
L = K.layers

PREACTIVATION = 'relu'


def preact_conv2D(inputs, k=3, filters=64, norm="instance"):
    if norm is "instance":
        outputs = InstanceNormalization()(inputs)
    else:
        outputs = L.BatchNormalization()(inputs)
    outputs = L.Activation(PREACTIVATION)(outputs)
    return L.Conv2D(filters, kernel_size=(k, k), padding='same')(outputs)

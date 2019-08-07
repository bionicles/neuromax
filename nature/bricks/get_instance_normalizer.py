import tensorflow_addons as tfa
import tensorflow as tf

K = tf.keras


def get_instance_normalizer(shape):
    return K.Sequential([
        K.Input(shape),
        tfa.layers.InstanceNormalization()])

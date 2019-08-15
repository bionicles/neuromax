import tensorflow_probability as tfp
import tensorflow as tf

tfpl = tfp.layers
K = tf.keras


def get_mlp(input_shape, units_list):
    layers = [K.Input(input_shape)]
    for units, activation in units_list:
        layer = tfpl.DenseReparameterization(units, activation)
        layers.append(layer)
    return K.Sequential(layers)

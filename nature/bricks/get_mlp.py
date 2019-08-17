import tensorflow_probability as tfp
import tensorflow as tf

tfpl = tfp.layers
K = tf.keras

UNITS, FN, LAYERS = 32, "tanh", 2


def get_mlp(input_shape, layer_list=None,
            last_layer=None):
    if layer_list is None:
        if last_layer is None:
            layer_list = [(UNITS, FN) for _ in range(LAYERS)]
        else:
            layer_list = [(UNITS, FN) for _ in range(LAYERS-1)]
            layer_list.append(last_layer)
    layers = [K.Input(input_shape)] + [
        tfpl.DenseReparameterization(units, activation)
        for units, activation in layer_list]
    return K.Sequential(layers)

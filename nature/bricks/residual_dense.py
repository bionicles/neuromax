import tensorflow as tf

from .layers.conv import get_deconv_2D_out, get_conv_2D_out, get_conv_1D_out
from .layers.dense import get_dense_out
from .preact import get_norm_preact_out

L = tf.keras.layers

DENSE_LAYER_FN = get_conv_2D_out
DENSE_KERNEL_SIZE = 4
DENSE_PADDING = "same"
DENSE_N_LAYERS = 2
DENSE_FILTERS = 4
DENSE_UNITS = 32

RESIDUAL_LAYER_FN = get_conv_2D_out
RESIDUAL_KERNEL_SIZE = 4
RESIDUAL_PADDING = "same"
RESIDUAL_N_LAYERS = 2
RESIDUAL_FILTERS = 4
RESIDUAL_UNITS = 32

NORM = None
FN = None


def get_dense_block_out(
        agent, id, input, kernel_size=DENSE_KERNEL_SIZE, units=DENSE_UNITS,
        n_layers=DENSE_N_LAYERS, layer_fn=DENSE_LAYER_FN,
        filters=DENSE_FILTERS, padding=DENSE_PADDING):
    new_features = []
    for i in range(n_layers):
        out = get_norm_preact_out(agent, id, input)
        if layer_fn in [get_conv_2D_out, get_deconv_2D_out, get_conv_1D_out]:
            out = layer_fn(
                agent, id, input,
                kernel_size=kernel_size, filters=filters, padding=padding)
        elif layer_fn is get_dense_out:
            out = layer_fn(agent, id, input, units=units)
        new_features.append(out)
        input = tf.concat([input, out], axis=-1)
    return tf.concat(new_features, axis=-1)


def get_residual_block_out(
        agent, id, input, layer_fn=RESIDUAL_LAYER_FN, units=RESIDUAL_UNITS,
        n_layers=RESIDUAL_N_LAYERS, kernel_size=RESIDUAL_KERNEL_SIZE,
        filters=RESIDUAL_FILTERS, norm=NORM, fn=FN, padding=RESIDUAL_PADDING):
    out = get_norm_preact_out(agent, id, input, norm=norm, fn=fn)
    if layer_fn in [get_conv_2D_out, get_deconv_2D_out, get_conv_1D_out]:
        out = layer_fn(
            agent, id, input, filters=filters, padding=padding)
    elif layer_fn is get_dense_out:
        out = layer_fn(agent, id, input, units=units)
    for n in range(n_layers - 1):
        out = get_norm_preact_out(agent, id, out, norm=norm, fn=fn)
        if layer_fn in [get_conv_2D_out, get_deconv_2D_out, get_conv_1D_out]:
            out = layer_fn(
                agent, id, input, filters=filters, padding=padding)
        elif layer_fn is get_dense_out:
            out = layer_fn(agent, id, input, units=units)
    return L.Add()([input, out])

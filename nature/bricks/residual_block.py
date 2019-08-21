import tensorflow as tf

from nature import use_norm_preact, use_conv_2D, use_layer, use_input
from tools import make_uuid

K = tf.keras
L = K.layers


LAYER_FN = use_conv_2D
KERNEL_SIZE = 4
PADDING = "same"
N_LAYERS = 2
FILTERS = 4
UNITS = 32
NORM = 'instance'
FN = "swish"


def use_residual_block(
        agent, id, input, layer_fn=LAYER_FN, units=UNITS,
        n_layers=N_LAYERS, kernel_size=KERNEL_SIZE,
        filters=FILTERS, norm=NORM, fn=FN, padding=PADDING,
        return_brick=False):
    # we update the id
    id = make_uuid([id, "residual_block"])
    # we make a model
    input_layer = use_input(agent, id, input)
    out = use_norm_preact(agent, id, input_layer, norm=norm, fn=fn)
    out = use_layer(agent, id, layer_fn, out, units, return_brick=False)
    for n in range(n_layers - 1):
        out = use_norm_preact(agent, id, out, norm=norm, fn=fn)
        out = use_layer(agent, id, layer_fn, out, units, return_brick=False)
    out = L.Add()([input_layer, out])
    model = K.Model(input_layer, out)
    parts = dict(model=model)
    # we make a function to use it
    call = model.call
    return agent.pull_brick(parts)

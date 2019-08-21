import tensorflow as tf

from nature import use_norm_preact, use_layer, use_conv_2D, use_input
from tools import make_uuid

K = tf.keras
L = K.layers

LAYER_FN = use_conv_2D
KERNEL_SIZE = 4
PADDING = "same"
N_LAYERS = 2
FILTERS = 4
UNITS = 32


def use_dense_block(
        agent, id, input, return_brick=False,
        kernel_size=KERNEL_SIZE, units=UNITS, n_layers=N_LAYERS,
        layer_fn=LAYER_FN, filters=FILTERS, padding=PADDING):
    # we update the id
    id = make_uuid([id, "dense_block"])

    # we build the parts
    input_layer = use_input(agent, id, input)
    new_features = []
    out = None
    for i in range(n_layers):
        x = input_layer if out is None else out
        out = use_norm_preact(agent, f"{id}_{i}", x)
        out = use_layer(
            agent, f"{id}_{i}", layer_fn,
            filters=filters, units=units, input=out)
        new_features.append(out)
        input = tf.concat([input, out], axis=-1)
    out = tf.concat(new_features, axis=-1)
    model = K.Model(input_layer, out)
    parts = dict(model=model)
    # we make the function
    call = model.call
    return agent.pull_brick(id, parts, call, input, return_brick)

from .dense import get_dense_out


def get_layer(agent, id, input, layer_fn, d_in):
    if layer_fn in [get_dense_out, None]:
        _, layer = layer_fn(agent, id, input, units=d_in, return_brick=True)
    else:
        _, layer = layer_fn(agent, id, input, filters=d_in, return_brick=True)
    return layer

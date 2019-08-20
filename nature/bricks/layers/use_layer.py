from nature import use_dense


def use_layer(
        agent, id, layer_fn,
        d_out=None, filters=None, units=None, input=None, return_brick=True):
    """get a layer with a desired output dimension"""
    if layer_fn in [use_dense, None]:
        units = d_out if d_out else units
        return use_dense(
            agent, id, input, units=units, return_brick=return_brick)
    else:
        filters = d_out if d_out else filters
        return layer_fn(
            agent, id, input, filters=filters, return_brick=return_brick)

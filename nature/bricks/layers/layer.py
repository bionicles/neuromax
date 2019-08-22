from .dense import use_dense


def use_layer(layer_fn, d_out):
    """get a layer with a desired output dimension"""
    if layer_fn in [use_dense, None]:
        return use_dense(units=d_out)
    else:
        return layer_fn(filters=d_out)

from nature import use_linear


def use_layer(layer_fn, units_or_filters):
    """
    get a layer with a desired output dimension
    args:
        layer_fn: callable to build the layer
        units_or_filters: int for linear units or dense filters
    """
    if layer_fn in [use_linear, None]:
        return use_linear(units=units_or_filters)
    else:
        return layer_fn(filters=units_or_filters)

from nature import Linear


def Layer(layer, units_or_filters):
    """
    get a layer with a desired output dimension
    args:
        layer: callable to build the layer
        units_or_filters: int for linear units or dense filters
    """
    if layer in [Linear, None]:
        return Linear(units=units_or_filters)
    else:
        return layer(filters=units_or_filters)

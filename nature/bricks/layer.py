from nature import FC, AllAttention, Quadratic


def Layer(units_or_filters, layer_fn=None, keepdim=False):
    """
    get a layer with a desired output dimension
    args:
        layer_fn: callable to build the layer
        units_or_filters: int for linear units or dense filters
    """
    if layer_fn is AllAttention:
        return layer_fn(keepdim=keepdim)
    if layer_fn is None:
        layer_fn = FC
    return layer_fn(units=units_or_filters)

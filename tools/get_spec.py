from attrdict import AttrDict

# should we delete this and just use AttrDict?


def get_spec(shape=None, format=None, n=None, variables=None, add_coords=None):
    """
    Build an AttrDict for an input or output tensor

    Args:
        format: a string descriptor of the tensor's type ["onehot", "ragged"]
    """
    spec = AttrDict({})
    if shape is not None:
        spec["shape"] = shape
    if format is not None:
        spec["format"] = format
    if n is not None:
        spec["n"] = n
    if variables is not None:
        spec["variables"] = variables
    if add_coords is not None:
        spec["add_coords"] = True
    return spec

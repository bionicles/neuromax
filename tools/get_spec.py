from attrdict import AttrDict

# should we delete this and just use AttrDict?


def get_spec(shape=None, format=None, n=None, sensor_type=None):
    spec = AttrDict({})
    if shape is not None:
        spec["shape"] = shape
    if format is not None:
        spec["format"] = format
    if n is not None:
        spec["n"] = n
    if sensor_type is not None:
        spec["sensor_type"] = sensor_type
    return spec

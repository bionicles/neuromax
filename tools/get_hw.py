def get_hw(image_or_tuple):
    if hasattr(image_or_tuple, "shape"):
        shape = image_or_tuple.shape
    else:
        shape = image_or_tuple
    if len(shape) is 4:
        return shape[1], shape[2]
    else:
        return shape[0], shape[1]

from functools import reduce
from operator import mul
from tensorflow import is_tensor

from .log import log


def get_size(shape):
    log('get_size for', shape)
    if isinstance(shape, int):
        return shape
    if len(shape) is 1:
        return shape[0]
    if None in shape:
        raise Exception("cannot get size for shape:", shape)
    if is_tensor(shape):
        if hasattr(shape, "shape"):
            shape = shape.shape
        shape = tuple(shape)
    size = reduce(mul, shape)
    log(list(shape), "is size", int(size))
    return size

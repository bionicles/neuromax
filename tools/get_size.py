from functools import reduce
from operator import mul
from tensorflow import is_tensor

from .log import log


def get_size(shape):
    if isinstance(shape, int):
        return shape
    if None in shape:
        raise Exception("cannot get size for shape:", shape)
    if is_tensor(shape):
        if hasattr(shape, "shape"):
            shape = shape.shape
        shape = tuple(shape)
    size = reduce(mul, shape)
    log(list(shape), "is size", int(size))
    return size

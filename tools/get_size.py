from functools import reduce
from operator import mul

from .log import log


def get_size(shape):
    if None in shape:
        raise Exception("cannot get size for shape:", shape)
    size = reduce(mul, shape)
    log(list(shape), "is size", int(size))
    return size

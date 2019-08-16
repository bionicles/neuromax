from functools import reduce
from operator import mul


def get_size(shape):
    print("get size for", shape)
    if None in shape:
        raise Exception("cannot get size for shape:", shape)
    size = reduce(mul, shape)
    print("got size of", size)
    return size

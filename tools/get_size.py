from functools import reduce
from operator import mul
import tensorflow as tf


def get_size(shape):
    print("get size for", shape)
    if None in shape:
        raise Exception("cannot get size for shape:", shape)
    return reduce(mul, shape)

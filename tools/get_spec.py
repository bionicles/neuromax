# should we delete this and just use AttrDict?
from attrdict import AttrDict
import numpy as np

from tools import get_size


def get_spec(shape=None, format=None, n=None, variables=[],
             low=None, high=None):
    """
    Build an AttrDict for an input or output tensor

    Args:
        format: a string descriptor of the tensor's type ["onehot", "ragged"]
        shape: tensor shape
        size: int dimensionality
        n: number of discrete values
        variables: list of (name, position) for shape variables in input
        add__coords: boolean to determine if we should concat coordinates
        low: number for the bottom of the range of possible values
        high: number for the top of the range of possible values
    """
    spec = AttrDict({})
    if shape:
        if format is "onehot":
            shape = (n,)
        spec.shape = shape
    if format:
        spec.format = format
    if n:
        spec.n = n
    if variables:
        spec.variables = variables
    if low and low is not -np.inf:
        spec.low = low
    if high and high is not np.inf:
        spec.high = high
    spec.rank = len(spec.shape)
    try:
        spec['size'] = get_size(shape)
    except Exception as e:
        print("get_spec failed to get size", e)
    return spec

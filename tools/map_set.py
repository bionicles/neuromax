from itertools import combinations as sets
import tensorflow as tf

np = None


def map_sets_tf(tensor, set_size, fn, pool):
    """
    Map_fn FN over SET_SIZE sets in TENSOR & POOL results

    Args:
        tensor: tf.Tensor from which we draw sets
        set_size: integer number of elements from tensor upon which we operate
        fn: callable to apply to each set of set_size elements in tensor
        pool: callable to apply to the results of mapping fn over sets

    Returns:
        pooled result of fn mapped over set_size sets in tensor

    Roughly:
        pool(map(fn, sets(iterable, set_size)))
    """
    if np is None:
        import numpy as np
    indices = tf.flatten(np.indices(tensor.shape))
    return pool(
        tf.map_fn(
            lambda index_set: fn(tf.gather_nd(tensor, index_set)),
            list(sets(indices, set_size))
            )
        )


def map_sets(iterable, set_size, fn, pool):
    """
    Map FN over SET_SIZE sets in ITERABLE & POOL results

    Args:
        iterable: list from which we draw sets
        set_size: integer number of elements from tensor to be applied
        fn: callable to apply to each set of set_size elements in tensor
        pool: callable to apply to the results of mapping fn over sets

    Returns:
        pooled result of fn mapped over set_size sets in iterable

    Roughly:
        pool(map(fn, sets(iterable, set_size)))
    """
    return pool(
        map(
            lambda set: fn(set),
            sets(iterable, set_size)
            )
        )

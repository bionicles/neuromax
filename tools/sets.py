from itertools import combinations


def sets(elements, set_size):
    """Return all sets of elements of set_size
    Args:
        elements: iterable
        set_size: number of elements per set
    """
    return combinations(elements, set_size)

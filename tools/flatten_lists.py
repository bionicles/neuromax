from itertools.chain import from_iterable


def flatten_lists(lists):
    return list(from_iterable(lists))

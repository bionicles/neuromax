from attrdict import AttrDict


def make_attrdict(*args, local_dict=None):
    assert local_dict is not None and isinstance(local_dict, dict)
    if isinstance(args[0], list):
        args = args[0]
    return AttrDict({k: local_dict[k] for k in args})

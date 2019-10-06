from attrdict import AttrDict


def map_attrdict(fn, attr_dict, *args, **kwargs):
    """ apply a function to all key, value pairs in an AttrDict """
    return AttrDict([fn(k, v, *args, **kwargs) for k, v in attr_dict.items()])

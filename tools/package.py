from attrdict import AttrDict
from sorcery import dict_of


def package(*args):
    return AttrDict(dict_of(args))

# neuromax.py - why?: config + train
from mol import PyMolEnv
from a2g import train


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


config = AttrDict({
    # env
    "MAX_UNDOCK_DISTANCE": 100,
    "MIN_UNDOCK_DISTANCE": 10,
    "MAX_STEPS_IN_UNDOCK": 3,
    "MIN_STEPS_IN_UNDOCK": 2,
    "STOP_LOSS_MULTIPLE": 10,
    "ATOM_JIGGLE": 1,
})

env = PyMolEnv(config)
train(env)

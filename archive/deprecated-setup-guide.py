# to register custom env we must edit gym source
#
# 1. in /home/<NAME>/anaconda3/envs/<ENV_NAME>/lib/python3.6/site-packages/gym
#
# 2. add /envs/mol/mol.py with custom env file mol.py
#
# 3. add /envs/mol/__init__.py with "from gym.envs.mol.mol import PyMolEnv"
#
# 4. add /spaces/array.py with custom space file array.py
#
# 5. edit /spaces/__init__.py to add from gym.spaces.array import Array
# 5.2 also add "Array" to the list called "__all__"

# 6. change spinningup/spinup/algos/td3/td3.py line 137, 138 to [1] not [0]
# 6,2 change algos/td3/core.py line 29 to [1] not [0]
#
# 6. add the code at the bottom to /envs/__init__.py:


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


config = AttrDict({
    "MAX_UNDOCK_DISTANCE": 100,
    "MIN_UNDOCK_DISTANCE": 10,
    "MAX_STEPS_IN_UNDOCK": 3,
    "MIN_STEPS_IN_UNDOCK": 2,
    "STOP_LOSS_MULTIPLE": 10,
    "ACTION_DIMENSION": 3,
    "ATOM_DIMENSION": 17,
    "NUM_EPISODES": 100,
    "NUM_STEPS": 100,
    "IMAGE_SIZE": 256,
    "ATOM_JIGGLE": 1,
})
kwargs = {"config": config}
register(
    id='PyMolEnv-v0',
    entry_point='gym.envs.mol:PyMolEnv',
    kwargs=kwargs
)

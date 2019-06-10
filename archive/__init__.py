# we register our custom environment with gym
from neuromax import config
from mol import PyMolEnv

kwargs = { "config": config }
gym.envs.register(
    id='PyMolEnv-v0',
    entry_point='mol:PyMolEnv',
    kwargs=kwargs
)

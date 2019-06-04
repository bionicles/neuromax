# neuromax.py - why?: config + train
from model import make_model
from mol import PyMolEnv

# we use a helper class for dict
class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

# we keep parameters together to save time
config = AttrDict({
    # env
    "MAX_UNDOCK_DISTANCE": 100,
    "MIN_UNDOCK_DISTANCE": 10,
    "MAX_STEPS_IN_UNDOCK": 4,
    "MIN_STEPS_IN_UNDOCK": 2,
    "STOP_LOSS_MULTIPLE": 10,
    "ACTION_DIMENSION": 3,
    "ATOM_DIMENSION": 17,
    "NUM_EPISODES": 100,
    "NUM_STEPS": 100,
    "IMAGE_SIZE": 256,
    "ATOM_JIGGLE": 1,
    # model
    "ACTIVATION": "tanh",
    "NUM_BLOCKS": 2,
    "NUM_LAYERS": 2,
    "UNITS": 500,
    "DROPOUT_RATE": 0.0,
    "CONNECT_BLOCK_AT": 4,
    "OUTPUT_SHAPE": 3
})

# we make the env and model
env = PyMolEnv(config)
model = Model(config)

# we run the training
for episode_number in config.NUM_EPISODES:
    observation = env.reset()
    done = false
    while not done:
        action = model.step(observation)
        observation, reward, done = env.step(action)

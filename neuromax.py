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
    "MAX_STEPS_IN_UNDOCK": 3,
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
    "NUM_BLOCKS": 6,
    "NUM_LAYERS": 2,
    "UNITS": 50,
    "DROPOUT": False,
    "DROPOUT_RATE": 0.1,
    "CONNECT_BLOCK_AT": 2,
    "OUTPUT_SHAPE": 3,
    "LOSS_FUNCTION": "mse",
    "OPTIMIZER": "adam"
})

# we make the env and model
env = PyMolEnv(config)
model = make_model(config)

# we run the training
for episode_number in range(config.NUM_EPISODES):
    observation = env.reset()
    done = False
    while not done:
        action = model.predict(observation)
        observation, reward, done, info = env.step(action)

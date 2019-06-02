# neuromax.py - why?: config + train
class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
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
    "NUM_BLOCKS": 2,
    "NUM_LAYERS": 2
})
# make the models
env = PyMolEnv(config)
model = Model(config)
# one protein
def run_episode(episode_number):
    observation = env.reset()
    done = false
    while not done:
        run_step()
# one step
def run_step():
    action = model.step(observation)
    observation, reward, done = env.step(action)
# run the training
for episode_number in config.NUM_EPISODES:
    run_episode(episode_number)

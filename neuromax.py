from attrdict import AttrDict
import tensorflow as tf
import gym

from nature.agent import Agent

from nurture.clevr.clevr import read_clevr_dataset, run_clevr_task
from nurture.mol.mol import read_mol_dataset, run_mol_task
from nurture.gym.gym import get_env_io, run_env_task

from helpers import map_attrdict, log

WORD_VECTOR_LENGTH = 300

MAX_LOOPS = 100
tasks = AttrDict({
    "MountainCar-v0": {
        "type": "env",
        "env": gym.make("MountainCar-v0"),
        "runner": run_env_task
       },
    "mol": {
        "type": "dataset",
        "inputs": [AttrDict({"shape": ("n_atoms", 10), "dtype": tf.float32})],
        "outputs": [AttrDict({"shape": ("n_atoms", 3), "dtype": tf.float32})],
        "dataset": read_mol_dataset(),
        "runner": run_mol_task
        },
    "clevr": {
        "type": "dataset",
        "name": "clevr",
        "inputs": [
            AttrDict({"shape": (480, 320, 3), "dtype": tf.float32}),
            AttrDict({"shape": (None, WORD_VECTOR_LENGTH), "dtype": tf.float32})],
        "outputs": [AttrDict({"shape": (28, ), "dtype": tf.int32})],
        "dataset": read_clevr_dataset(),
        "runner": run_clevr_task
    }
})
tasks = map_attrdict(get_env_io, tasks)

human_level, converged, loops = False, False, 0
agent = Agent(tasks)

while not (human_level or converged):
    results, human_level, converged = agent.train()
    log(f"human_level: {human_level}")
    log(f"converged: {converged}")
    log(f"results: {results}")
    loops += 1
    if loops > MAX_LOOPS:
        break

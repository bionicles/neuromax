from attrdict import AttrDict
import gym

from nurture.clevr.clevr import read_clevr_dataset, run_clevr_task
from nurture.mol.mol import read_mol_dataset, run_mol_task
from nurture.gym.gym import get_env_io_specs, run_env_task
from nature.agent import Agent

from tools import map_attrdict, log, get_spec

WORD_VECTOR_LENGTH = 300
MAX_LOOPS = 100


tasks = AttrDict({
    "MountainCar-v0": {
        "type": "env",
        "env": gym.make("MountainCar-v0"),
        "runner": run_env_task,
        "episodes_per_session": 5},
    "mol": {
        "type": "dataset",
        "inputs": [get_spec(shape=("n_atoms", 10), sensor_type="lstm")],
        "outputs": [get_spec(shape=("n_atoms", 3), format="ragged")],
        "dataset": read_mol_dataset(),
        "runner": run_mol_task,
        "examples_per_episode": 5},
    "clevr": {
        "type": "dataset",
        "name": "clevr",
        "inputs": [get_spec(shape=(480, 320, 3), sensor_type="image"),
                   get_spec(shape=(None, WORD_VECTOR_LENGTH),
                            sensor_type="lstm")],
        "outputs": [get_spec(shape=(28, ), format="onehot")],
        "dataset": read_clevr_dataset(),
        "runner": run_clevr_task,
        "examples_per_episode": 5}})

# we build env I/O specs for gym tasks:
tasks = map_attrdict(get_env_io_specs, tasks)

human_level, converged, loops = False, False, 0
agent = Agent(tasks)

for loop_number in range(MAX_LOOPS):
    human_level, converged = agent.train()
    log(f"loop: {loop_number}")
    log(f"human_level: {human_level}")
    log(f"converged: {converged}")
    if human_level or converged:
        break

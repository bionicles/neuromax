from attrdict import AttrDict
import gym

from nurture.clevr.clevr import read_clevr_dataset, run_clevr_task
from nurture.mol.mol import read_mol_dataset, run_mol_task
from nurture.gym.gym import get_env_io_specs, run_env_task
from nature.agent import Agent

from tools import map_attrdict, get_spec

WORD_VECTOR_SIZE = 300
MAX_LOOPS = 100


tasks = AttrDict({
    "MountainCar-v0": {
        "type": "env",
        "env": gym.make("MountainCar-v0"),
        "runner": run_env_task},
    "mol": {
        "type": "dataset",
        "inputs": [get_spec(shape=("n_atoms", 10), format="ragged")],
        "outputs": [get_spec(shape=("n_atoms", 3), format="ragged")],
        "dataset": read_mol_dataset(),
        "runner": run_mol_task},
    "clevr": {
        "type": "dataset",
        "inputs": [get_spec(shape=(480, 320, 4), format="image"),
                   get_spec(shape=(None, WORD_VECTOR_SIZE), format="ragged")],
        "outputs": [get_spec(shape=(28, ), format="onehot")],
        "dataset": read_clevr_dataset(),
        "runner": run_clevr_task}})

# we build env I/O specs for gym tasks:
tasks = map_attrdict(get_env_io_specs, tasks)

agent = Agent(tasks)

results = [agent.train() for _ in range(MAX_LOOPS)]

print("results")
print(results)

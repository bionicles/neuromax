from attrdict import AttrDict
import gym

from nurture.clevr.clevr import read_clevr_dataset, run_clevr_task
from nurture.mol.mol import read_mol_dataset, run_mol_task
from nurture.gym.gym import get_env_io_specs, run_env_task
from nature.agent import Agent

from tools.map_attrdict import map_attrdict
from tools.get_spec import get_spec

WORD_VECTOR_SIZE = 300
MAX_LOOPS = 100

print("neuromax.py running")
tasks = AttrDict({
    "clevr": {
        "type": "dataset",
        "inputs": [get_spec(
            shape=(320, 480, 4),
            add_coords=True,
            format="image"
            ),
                   get_spec(
            shape=(None, WORD_VECTOR_SIZE),
            add_coords=True,
            format="ragged",
            variables=[("n_words", 0, -2)]
            )],
        "outputs": [get_spec(shape=(28, ), format="onehot")],
        "dataset": read_clevr_dataset(),
        "run_agent_on_task": run_clevr_task,
        "examples_per_episode": 10},
    "MountainCar-v0": {
        "type": "env",
        "env": gym.make("MountainCar-v0"),
        "run_agent_on_task": run_env_task,
        "examples_per_episode": 10},
    "mol": {
        "type": "dataset",
        "inputs": [get_spec(shape=(None, 10),
                            add_coords=True,
                            format="ragged",
                            variables=[("n_atoms", 0, -2)])],
        "outputs": [get_spec(shape=(None, 3),
                             format="ragged",
                             variables=[("n_atoms", 0, -2)])],
        "dataset": read_mol_dataset(),
        "run_agent_on_task": run_mol_task,
        "examples_per_episode": 10}
                 })

# we build env I/O specs for gym tasks:
print("preparing tasks:\n", tasks)
tasks = map_attrdict(get_env_io_specs, tasks)

agent = Agent(tasks)

results = [agent.train() for _ in range(MAX_LOOPS)]

print("results")
print(results)

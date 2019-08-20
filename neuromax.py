from attrdict import AttrDict
import tensorflow as tf
import logging

from nurture import read_clevr_dataset, run_clevr_task, get_env_io_specs
from nature import Agent

from tools import map_attrdict, get_spec

tf.get_logger().setLevel(logging.WARNING)

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
    # "mol": {
    #     "type": "dataset",
    #     "inputs": [get_spec(shape=(None, 10),
    #                         add_coords=True,
    #                         format="ragged",
    #                         variables=[("n_atoms", 0, -2)])],
    #     "outputs": [get_spec(shape=(None, 3),
    #                          format="ragged",
    #                          variables=[("n_atoms", 0, -2)])],
    #     "dataset_fn": read_mol_dataset,
    #     "run_agent_on_task": run_mol_task,
    #     "examples_per_episode": 10},
    # "MountainCar-v0": {
    #     "type": "env",
    #     "env": gym.make("MountainCar-v0"),
    #     "run_agent_on_task": run_env_task,
    #     "examples_per_episode": 10},
                 })

# we build env I/O specs for gym tasks:
print("preparing tasks.\n", )
tasks = map_attrdict(get_env_io_specs, tasks)
[[print(tk, k, v) for k, v in tv.items()] for tk, tv in tasks.items()]

agent = Agent(tasks)

results = [agent.train() for _ in range(MAX_LOOPS)]

print("results")
print(results)

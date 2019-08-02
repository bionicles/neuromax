# nurture.py - bion and kamel - august 2019
# why?: solve many tasks with 1 training loop
from attrdict import AttrDict
import tensorflow as tf

from nurture.clevr.clevr import run_clevr_episode
from nurture.mol.mol import run_mol_episode

trainers = AttrDict({
    "clevr": run_clevr_episode,
    "mol": run_mol_episode
})


def get_trainer(task):
    if task.type == "env":
        return run_gym_episode
    return trainers[task.name]


def handle_nones(X):
    return [tf.zeros((1,) + input_shapes[i])
            if x is None else x
            for i, x in enumerate(X)]
    
def run_agent_on_env(agent, env_name, n_episodes):
    env = gym.make(env_name)
    for _ in range(n_episodes):
        observation = env.reset()
        done = False
        while not done:
            action = agent(observation)
            observation, loss, done, _ = env.step(action)
            

def train(agent, optimizer, tasks):
    # we get the trainers
    # we run the tasks
    raise NotImplementedError

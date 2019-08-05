from attrdict import AttrDict
import tensorflow as tf


def space2dict(space):
    type_name = type(space).__name__
    space_dict = AttrDict({
        "object": space,
        "type_name": type_name,
    })
    if type_name == "Box":
        space_dict.shape = space.shape
        space_dict.dtype = tf.float32
        space_dict.low = space.low
        space_dict.high = space.high
    if type_name == "Discrete":
        space_dict.dtype = tf.int32
        space_dict.shape = (1)
        space_dict.high = space.n - 1
        space_dict.low = 0
    return space_dict


def get_env_io(task):
    if task.type is "env":
        task.inputs = [space2dict(task.env.observation_space)]
        task.outputs = [space2dict(task.env.action_space)]


def run_env_task(agent, task, n_episodes):
    for _ in range(n_episodes):
        observation = task.env.reset()
        done = False
        while not done:
            codes, reconstructions, predictions, actions = agent(observation)
            new_observation, maybe_loss_or_reward, done, _ = task.env.step(actions)
        agent.memorize((
            observation, codes, reconstructions, predictions, actions,
            new_observation, maybe_loss_or_reward, done))

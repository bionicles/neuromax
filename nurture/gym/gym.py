from attrdict import AttrDict
import tensorflow as tf

from tools.compute_surprise import compute_surprise
from tools.sum_entropy import sum_entropy
from tools.compute_kl import compute_kl
from tools.get_spec import get_spec


def space2spec(space):
    type_name = type(space).__name__
    if type_name == "Discrete":
        spec = get_spec(format="discrete", n=space.n)
    if type_name == "Box":
        spec = get_spec(format="box", shape=space.shape,
                        low=space.low, high=space.high)
    return spec


def get_env_io_specs(task_key, task_dict):
    print("get_env_io_specs", task_key, task_dict)
    task_dict = AttrDict(task_dict)
    if task_dict.type is "env":
        task_dict.inputs = [space2spec(task_dict.env.observation_space)]
        task_dict.outputs = [space2spec(task_dict.env.action_space)]
    return task_key, task_dict


def run_env_task(agent, task_key, task_dict):
    prior_state_predictions = None
    model = agent.models[task_key]
    total_free_energy = 0.
    env = task_dict.env
    for _ in range(task_dict.episodes_per_session):
        observation = env.reset()
        inputs = [observation]
        done = False
        while not done:
            with tf.GradientTape() as tape:
                normies, codes, reconstructions, state_predictions, \
                    loss_prediction, actions = model(inputs)
                action = actions.sample()
                new_observation, reward, done, _ = env.step(action)
                loss = reward * -1
                reconstruction_surprise = compute_surprise(
                    reconstructions, normies)
                if prior_state_predictions:
                    state_surprise = compute_kl(prior_state_predictions, codes)
                else:
                    state_surprise = 0.
                prior_state_predictions = state_predictions
                loss_surprise = compute_surprise([loss_prediction], [loss])
                surprise = reconstruction_surprise + state_surprise + loss_surprise
                freedom = sum_entropy(actions)
                free_energy = loss + surprise - freedom
            gradients = tape.gradient([free_energy, model.losses], model.trainable_variables)
            agent.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            inputs = [new_observation]
            total_free_energy = total_free_energy + free_energy
    return total_free_energy

from attrdict import AttrDict
import tensorflow as tf

from tools.get_onehot import get_onehot
from tools.get_spec import get_spec


def run_env_task(agent, task_key, task_dict):
    model = task_dict.model
    total_free_energy = 0.
    env = task_dict.env
    onehot_task_key = get_onehot(task_key, list(agent.tasks.keys()))
    for _ in range(task_dict.episodes_per_session):
        prior_loss_prediction = 0.
        prior_code_prediction = tf.zeros(
            agent.compute_code_shape(task_dict))
        observation = env.reset()
        inputs = [onehot_task_key, 0., observation]
        done = False
        while not done:
            with tf.GradientTape() as tape:
                normies, code, actions = model(inputs)
                code_prediction, loss_prediction, reconstructions, forces = agent.unpack_actions(task_key, actions)
                action_samples = [action.sample() for action in actions]
                rescale_boxes(action_samples, task_dict)
                action_samples = action_samples[0] if len(task_dict.outputs) is 1 else action_samples
                new_observation, reward, done, _ = env.step(action_samples)
                loss = reward * -1
                free_energy = agent.compute_free_energy(
                    loss=loss, prior_loss_prediction=prior_loss_prediction,
                    normies=normies, reconstructions=reconstructions,
                    code=code, prior_code_prediction=prior_code_prediction,
                    actions=actions
                )
            gradients = tape.gradient([free_energy, model.losses], model.trainable_variables)
            agent.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            inputs = [new_observation]
            total_free_energy = total_free_energy + free_energy
    return total_free_energy


def rescale_boxes(action_samples, task_dict):
    rescaled_action_samples = []
    for action_sample, out_spec in zip(action_samples, task_dict.outputs):
        out_keys = task_dict.outputs[0].format.keys()
        if "low" in out_keys and "high" in out_keys:
            action_sample = action_sample * out_spec.high - out_spec.low
        else:
            action_sample = action_sample
        rescaled_action_samples.append(action_sample)
    return rescaled_action_samples


def space2spec(space):
    type_name = type(space).__name__
    if type_name == "Discrete":
        spec = get_spec(format="discrete", n=space.n,
                        shape=(space.n,))
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

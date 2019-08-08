import tensorflow as tf

from tools import get_spec, compute_surprise, compute_freedom, compute_kl


def space2spec(space):
    type_name = type(space).__name__
    if type_name == "Discrete":
        spec = get_spec(format="discrete", n=space.n, sensor_type="dense")
    if type_name == "Box":
        spec = get_spec(shape=space.shape, format="box", sensor_type="dense")
    return spec


def get_env_io_specs(task):
    if task.type is "env":
        task.inputs = [space2spec(task.env.observation_space)]
        task.outputs = [space2spec(task.env.action_space)]


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
                freedom = compute_freedom(actions)
                free_energy = loss + surprise - freedom
            gradients = tape.gradient([free_energy, model.losses], model.trainable_variables)
            agent.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            inputs = [new_observation]
            total_free_energy = total_free_energy + free_energy
    return total_free_energy

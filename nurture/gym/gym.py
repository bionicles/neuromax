from tools import get_spec


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

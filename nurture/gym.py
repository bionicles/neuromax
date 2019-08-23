from attrdict import AttrDict
import random
import gym

from tools.get_onehot import log, get_spec


def get_spec_for_space(space):
    type_name = type(space).__name__
    if type_name == "Discrete":
        return get_spec(format="discrete", n=space.n, shape=(space.n,))
    if type_name == "Box":
        return get_spec(
            format="box", shape=space.shape, low=space.low, high=space.high)
    raise Exception(space, "not Discrete or Box ... get_spec_for_space")


def prepare_env():
    env_list = [spec.id for spec in gym.envs.registry.all()]
    key = random.choice(env_list)
    try:
        env = gym.make(key)
        in_specs = [get_spec_for_space(env.observation_space)]
        out_specs = [get_spec_for_space(env.action_space)]
        return AttrDict(is_data=False, key=key, env=env,
                        in_specs=in_specs, out_specs=out_specs)
    except Exception as e:
        log(e, color="red")


def rescale_boxes(action_samples, task_dict):
    rescaled_action_samples = []
    for action_sample, out_spec in zip(action_samples, task_dict.outputs):
        out_keys = task_dict.outputs[0].format.keys()
        action_sample = action_sample
        if "low" in out_keys and "high" in out_keys:
            action_sample = action_sample * out_spec.high - out_spec.low
        rescaled_action_samples.append(action_sample)
    return rescaled_action_samples

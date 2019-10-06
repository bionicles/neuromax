from attrdict import AttrDict
# import random
import gym

from tools import log, get_spec


def get_spec_for_space(space):
    log("get_spec_for_space", space, color="red")
    type_name = type(space).__name__
    log("type_name", type_name, color="red")
    if type_name == "Discrete":
        spec = get_spec(format="discrete", n=space.n, shape=(space.n,))
    if type_name == "Box":
        spec = get_spec(
            format="box", shape=space.shape, low=space.low, high=space.high)
    log("spec", spec, color="red")
    return spec


def prepare_env(agent):
    log("prepare_env", color="red")
    # env_list = [spec.id for spec in gym.envs.registry.all()]
    # key = random.choice(env_list)
    key = "MountainCarContinuous-v0"
    env = gym.make(key)
    in_specs = [get_spec_for_space(env.observation_space)]
    out_specs = [get_spec_for_space(env.action_space)]
    return AttrDict(
        is_data=False, key=key, env=env,
        in_specs=in_specs, out_specs=out_specs)


def rescale_boxes(action_samples, task_dict):
    rescaled_action_samples = []
    for action_sample, out_spec in zip(action_samples, task_dict.outputs):
        out_keys = task_dict.outputs[0].format.keys()
        action_sample = action_sample
        if "low" in out_keys and "high" in out_keys:
            action_sample = action_sample * out_spec.high - out_spec.low
        rescaled_action_samples.append(action_sample)
    return rescaled_action_samples

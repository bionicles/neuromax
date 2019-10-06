from attrdict import AttrDict
from tools import get_spec_for_space


def organize_task(task_key, task_dict):
    print("get_env_io_specs", task_key, task_dict)
    task_dict = AttrDict(task_dict)
    if task_dict.type is "env":
        task_dict.inputs = [
            get_spec_for_space(task_dict.env.observation_space)]
        task_dict.outputs = [
            get_spec_for_space(task_dict.env.action_space)]
    return task_key, task_dict

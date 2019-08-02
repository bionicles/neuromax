# agent.py - handle multitask AI
from attrdict import AttrDict
import tensorflow as tf

from .bricks.interface import get_interface
from .graphmodel import GraphModel

K = tf.keras
L = K.layers


def map_attrdict(fn, attr_dict, *args, **kwargs):
    """ apply a function to all key, value pairs in an AttrDict """
    return AttrDict([fn(k, v, *args, **kwargs) for k, v in attr_dict.items()])


def map_enumerate(fn_list, arg_list, *args, **kwargs):
    """ apply a list of functions to a list of arguments """
    if callable(fn_list):
        return [fn_list(arg, *args, *kwargs) for arg in arg_list]
    elif len(fn_list) == len(arg_list):
        return [fn_list[i](arg_list[i], *args, **kwargs)
                for i in range(len(fn_list))]
    elif len(fn_list) is 1:
        return [fn_list[0](arg_list[i], *args, **kwargs)
                for i in range(len(arg_list))]
    else:
        raise Exception("map_enumerate needs 1 or N functions!")


def add_task_interfaces(task_key, task_dict, code_shape):
    """ adds a list of interface models for sensation and action to a task """
    task_dict.sensors = [
        get_interface(input_shape, code_shape)
        for input_shape in task_dict.inputs]
    task_dict.actuators = [
        get_interface(code_shape, output_shape)
        for output_shape in task_dict.outputs]
    return task_key, task_dict


class Agent:
    """ learns to map a set of sets of inputs onto a set of sets of outputs """

    def __init__(self, tasks, hp):
        self.tasks = map_attrdict(add_task_interfaces, tasks, hp.code_shape)
        self.shared_model = GraphModel(self.tasks, hp)

    def __call__(self, task_key, inputs):
        task_dict = self.tasks[task_key]
        sensations = map_enumerate(task_dict.sensors, inputs)
        judgments = self.shared_model(sensations)
        actions = map_enumerate(task_dict.actuators, judgments)
        return actions

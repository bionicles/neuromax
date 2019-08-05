# agent.py - handle multitask AI
from attrdict import AttrDict
import tensorflow as tf
import numpy as np
import random

from helpers import map_attrdict, map_enumerate
from .bricks.graphmodel import GraphModel
from .bricks.interface import Interface
from .bricks.coder import Coder


K = tf.keras
L = K.layers

NANOID_SIZE = 16


class Agent:
    """Entity which learns to solve tasks using bricks"""

    def __init__(self, tasks):
        self.code_rank = self.pull("code_rank", 1, 2)
        self.code_shape = tuple(self.pull("code_shape", ))
        self.tasks = map_attrdict(self.add_sensors_and_actuators, tasks)
        self.shared_model = GraphModel(self.tasks)
        self.shapes = AttrDict({})
        self.p = AttrDict({})

    def __call__(self, task_key, inputs):
        """
        Encode, reconstruct, predict, and decide for a task's input[s]

        Args:
        tkey: string key of the task
        inputs: a list of tensors
        """
        task_dict = self.tasks[task_key]
        codes, reconstructions = map_enumerate(task_dict.sensors, inputs)
        predictions = self.shared_model(codes)
        judgment = tf.concat([*codes, *predictions], 0)
        actions = map_enumerate(task_dict.actuators, judgment)
        return codes, reconstructions, predictions, actions

    def pull_numbers(self, pkey, a, b, step=1, n=1):
        """
        Provide numbers from range(min, max, step).
        If agent has pkey already, return its' value, else, sample a new value.
        if a and b are ints, return int, else return float

        Args:
        pkey: string key for this parameter
        a: low end of the range
        b: high end of the range
        step: distance between options
        n: number of numbers to pull
        """
        if pkey in self.p.keys():
            return self.p[pkey]
        if a == b:
            value = [min for _ in range(n)]
        if a > b:
            a, b = b, a
        elif isinstance(a, int) and isinstance(b, int):
            value = [random.randrange(a, b, step) for _ in range(n)]
        else:
            if step is 1:
                value = [random.uniform(a, b) for _ in range(n)]
            else:
                value = np.random.choice(np.arange(a, b, step),
                                         size=n).tolist()
        value = value if n > 1 else value[0]
        self.p[pkey] = value
        return value

    def pull_choices(self, pkey, options, n=1, distribution=None):
        """
        Choose from a set of options.
        If n=1, return 1 option, else return a list of n options
        If distribution=None (default), choose uniformly

        Args:
        pkey: unique key to share values
        options: list of possible values
        n: number of choices to return
        distribution: probability to return each option in options
        """
        if pkey in self.p.keys():
            return self.p[pkey]
        if distribution is None:
            value = random.sample(options, n)
        else:
            value = np.random.choice(options, n, p=distribution).tolist()
        self.p[pkey] = value
        return value

    def add_sensors_and_actuators(self, task_key, task_dict):
        """Add coder sensor bricks and interface actuator bricks to an agent"""
        task_dict.sensors = [
            Coder(self, input_shape, self.code_shape)
            for input_shape in task_dict.inputs]
        task_dict.actuators = [
            Interface(self, self.code_shape, output_shape)
            for output_shape in task_dict.outputs]
        return task_key, task_dict

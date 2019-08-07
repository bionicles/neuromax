# agent.py - handle multitask AI
from attrdict import AttrDict
import tensorflow as tf
import numpy as np
import random

from tools import map_attrdict, map_enumerate, get_spec, flatten_lists
from . import Brick


K = tf.keras
L = K.layers

MIN_CODE_ATOMS, MAX_CODE_ATOMS = 8, 32
MIN_CODE_SIZE, MAX_CODE_SIZE = 8, 32
EPISODES_PER_PRACTICE_SESSION = 5
CONVERGENCE_THRESHOLD = 0.01
N_LOOKBACK_STEPS = 5


class Agent:
    """Entity which learns to solve tasks using bricks"""

    def __init__(self, tasks):
        code_atoms = self.pull_numbers("code_atoms",
                                       MIN_CODE_ATOMS, MAX_CODE_ATOMS)
        code_size = self.pull_numbers("code_size",
                                      MIN_CODE_SIZE, MAX_CODE_SIZE)
        self.code_spec = get_spec(shape=(code_atoms, code_size))
        self.tasks = map_attrdict(self.register_shape_variables, tasks)
        self.tasks = map_attrdict(self.add_sensors_and_actuators, tasks)
        self.decide_n_in_n_out()
        self.shared_model = Brick(self.n_in, self.code_spec, self.n_out,
                                  agent=self, brick_type="GraphModel")
        self.parameters = AttrDict({})
        self.replay = []

    def decide_n_in_n_out(self):
        self.n_in = max([len(task_dict.inputs)
                         for task_key, task_dict in self.tasks.items()])
        self.n_out = max([len(task_dict.outputs)
                          for task_key, task_dict in self.tasks.items()])

    def register_shape_variables(self, task_key, task_dict):
        """Register shape variables in the task dictionary"""
        for input_number, in_spec in enumerate(task_dict.inputs):
            for dimension_number, dimension in enumerate(in_spec.shape):
                if isinstance(dimension, str):
                    task_dict.shape_variables[dimension] = AttrDict({
                        "path": [input_number, dimension_number],
                        "value": None,
                        })
                    task_dict.inputs[input_number].shape[dimension_number] = None
        return task_key, task_dict

    def add_sensors_and_actuators(self, task_key, task_dict):
        """Add coder sensors and interface actuators to an agent"""
        task_dict.sensors = [
            Brick(task_key, in_spec, self.code_spec, input_number,
                  agent=self, brick_type="Sensor")
            for input_number, in_spec in enumerate(task_dict.inputs)]
        task_dict.actuators = [
            Brick(self, task_key, self.code_spec, out_spec, output_number,
                  agent=self, brick_type="Sensor")
            for output_number, out_spec in enumerate(task_dict.outputs)]
        return task_key, task_dict

    def train(self):
        [[v.task_runner(self, k, v) for k, v in self.tasks.items()]
            for episode_number in range(EPISODES_PER_PRACTICE_SESSION)]
        return self.check_human_level(), self.check_convergence()

    def check_human_level(self):
        """True if N_LOOKBACK_STEPS have mean loss below task threshold"""
        return all([np.mean(
            flatten_lists(v.losses[N_LOOKBACK_STEPS:])
            ) < v.threshold for _, v in self.tasks.items()])

    def check_convergence(self):
        """True if N_LOOKBACK_STEPS have loss variance below CONVERGENCE_THRESHOLD"""
        return all([np.var(
                flatten_lists(v.losses[N_LOOKBACK_STEPS:])
                ) < CONVERGENCE_THRESHOLD for _, v in self.tasks.items()])

    def __call__(self, task_key, task_dict, inputs):
        """
        Encode, reconstruct, predict, and decide for a task's input[s]

        Args:
            tkey: string key of the task
            inputs: a list of tensors

        Returns: tuple of tensors
            normies, codes, reconstructions, predictions, actions
        """
        # we update shape variables
        if "shape_variables" in task_dict.keys():
            for dimension_key in task_dict.shape_variables.keys():
                input_number, dimension_number = \
                    task_dict.shape_variables[dimension_key].path
                task_dict.shape_variables[dimension_key].value = \
                    tf.shape(inputs[input_number])[dimension_number]
        self.current_task_dict = task_dict
        inputs_with_specs = zip(inputs, task_dict.inputs)
        normies, codes, reconstructions = map_enumerate(task_dict.sensors, inputs_with_specs)
        predictions = self.shared_model(codes)
        judgment = tf.concat([*codes, *predictions], 0)
        actions = map_enumerate(task_dict.actuators, judgment, task_dict)
        return (normies, codes, reconstructions, predictions, actions)

    def pull_numbers(self, pkey, a, b, step=1, n=1):
        """
        Provide numbers from range(min, max, step).

        Args:
            pkey: string key for this parameter
            a: low end of the range
            b: high end of the range
            step: distance between options
            n: number of numbers to pull

        Returns:
            maybe_number_or_numbers: a number if n=1, a list if n > 1
            an int if a and b are ints, else a float
        """
        if pkey in self.parameters.keys():
            return self.parameters[pkey]
        if a == b:
            maybe_number_or_numbers = [a for _ in range(n)]
        if a > b:
            a, b = b, a
        elif isinstance(a, int) and isinstance(b, int):
            maybe_number_or_numbers = [
                random.randrange(a, b, step) for _ in range(n)]
        else:
            if step is 1:
                maybe_number_or_numbers = [
                    random.uniform(a, b) for _ in range(n)]
            else:
                maybe_number_or_numbers = np.random.choice(
                    np.arange(a, b, step),
                    size=n).tolist()
        if n is 1:
            maybe_number_or_numbers = maybe_number_or_numbers[0]
        self.parameters[pkey] = maybe_number_or_numbers
        return maybe_number_or_numbers

    def pull_choices(self, pkey, options, n=1, distribution=None):
        """
        Choose from a set of options.
        Uniform sampling unless distribution is given

        Args:
            pkey: unique key to share values
            options: list of possible values
            n: number of choices to return
            distribution: probability to return each option in options

        Returns:
            maybe_choice_or_choices: string if n is 1, else a list
        """
        if pkey in self.parameters.keys():
            return self.parameters[pkey]
        if distribution is None:
            maybe_choice_or_choices = random.sample(options, n)
        else:
            maybe_choice_or_choices = np.random.choice(options, n, p=distribution).tolist()
        if n is 1:
            maybe_choice_or_choices = maybe_choice_or_choices[0]
        self.parameters[pkey] = maybe_choice_or_choices
        return maybe_choice_or_choices

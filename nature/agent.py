# agent.py - handle multitask AI
from attrdict import AttrDict
import tensorflow as tf
import numpy as np
import random

from tools import map_attrdict, map_enumerate, get_spec, get_onehot, get_size
from bricks import Interface, GraphModel


K = tf.keras
L = K.layers

MIN_CODE_ATOMS, MAX_CODE_ATOMS = 8, 32
MIN_CODE_CHANNELS, MAX_CODE_CHANNELS = 8, 32
EPISODES_PER_PRACTICE_SESSION = 5
# CONVERGENCE_THRESHOLD = 0.01
# N_LOOKBACK_STEPS = 5


class Agent:
    """Entity which learns to solve tasks using bricks"""

    def __init__(self, tasks):
        code_atoms = self.pull_numbers("code_atoms",
                                       MIN_CODE_ATOMS, MAX_CODE_ATOMS)
        code_channels = self.pull_numbers("code_channels",
                                          MIN_CODE_CHANNELS, MAX_CODE_CHANNELS)
        self.code_spec = get_spec(shape=(code_atoms, code_channels), format="code")
        self.code_spec.size = get_size(self.code_spec.shape)
        self.tasks = map_attrdict(self.register_shape_variables, tasks)
        self.tasks = map_attrdict(self.add_sensors_and_actuators, tasks)
        task_name_spec = get_spec(shape=(len(tasks)), format="onehot")
        task_name_sensor = Interface(self, "task_key", task_name_spec, self.code_spec)
        self.sensors.append(task_name_sensor)
        self.decide_n_in_n_out()
        self.shared_model = GraphModel(self)
        self.parameters = AttrDict({})
        self.replay = []

    def decide_n_in_n_out(self):
        self.n_in = max([len(task_dict.inputs)
                         for task_key, task_dict in self.tasks.items()])
        self.n_out = max([len(task_dict.outputs)
                          for task_key, task_dict in self.tasks.items()])

    @staticmethod
    def get_shape_var_id(task_id, input_number, dimension_number):
        return f"{task_id}_{input_number}_{dimension_number}"

    def register_shape_variables(self, task_key, task_dict):
        """Register shape variables in the task dictionary"""
        for input_number, in_spec in enumerate(task_dict.inputs):
            for dimension_number, dimension in enumerate(in_spec.shape):
                if isinstance(dimension, str):
                    shape_var_id = self.get_shape_var_id(
                        task_key, input_number, dimension_number)
                    self.parameters[shape_var_id] = None
                    task_dict.inputs[input_number].shape[dimension_number] = None
        return task_key, task_dict

    def add_sensors_and_actuators(self, task_key, task_dict):
        """Add interfaces to a task_dict"""
        task_dict.actuators = []
        task_dict.sensors = []
        for input_number, in_spec in enumerate(task_dict.inputs):
            encoder = Interface(self, task_key, in_spec, self.code_spec,
                                input_number=input_number)
            task_dict.sensors.append(encoder)
            decoder = Interface(self, task_key, self.code_spec, in_spec,
                                input_number=input_number)
            task_dict.actuators.append(decoder)
        for output_number, out_spec in enumerate(task_dict.outputs):
            actuator = Interface(self, task_key, self.code_spec, out_spec)
            task_dict.actuators.append(actuator)
        return task_key, task_dict

    def train(self):
        """Run EPISODES_PER_PRACTICE_SESSION episodes"""
        [[v.task_runner(self, k, v)
          for k, v in self.tasks.items()]
            for episode_number in range(EPISODES_PER_PRACTICE_SESSION)]

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
        task_input = get_onehot(task_key, self.tasks.keys())
        inputs = [task_input] + inputs
        codes = map_enumerate(task_dict.sensors, inputs)
        judgments = self.shared_model(codes)
        world_model = tf.concat([*codes, *judgments], 0)
        actions = map_enumerate(task_dict.actuators, world_model, task_dict)
        return codes, judgments, actions

    def pull_numbers(self, pkey, a, b, step=1, n=1):
        """
        Provide numbers from range(a, b, step).

        Args:
            pkey: string key for this parameter
            a: low end of the range
            b: high end of the range
            step: distance between options
            n: number of numbers to pull

        Returns:
            maybe_number_or_numbers: a number if n is 1, a list if n > 1
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

    # def check_human_level(self):
    #     """True if N_LOOKBACK_STEPS have mean loss below task threshold"""
    #     return all([np.mean(
    #         flatten_lists(v.losses[N_LOOKBACK_STEPS:])
    #         ) < v.threshold for _, v in self.tasks.items()])
    #
    # def check_convergence(self):
    #     """True if N_LOOKBACK_STEPS have loss variance below CONVERGENCE_THRESHOLD"""
    #     return all([np.var(
    #             flatten_lists(v.losses[N_LOOKBACK_STEPS:])
    #             ) < CONVERGENCE_THRESHOLD for _, v in self.tasks.items()])

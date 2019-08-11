# agent.py - handle multitask AI
from attrdict import AttrDict
import tensorflow as tf
import numpy as np
import random

from tools.map_enumerate import map_enumerate
from tools.map_attrdict import map_attrdict
from tools.get_onehot import get_onehot
from tools.get_size import get_size
from tools.get_spec import get_spec

from nature.bricks.graph_model import GraphModel
from nature.bricks.interface import Interface


K = tf.keras
L = K.layers

MIN_CODE_CHANNELS, MAX_CODE_CHANNELS = 8, 32
MIN_CODE_ATOMS, MAX_CODE_ATOMS = 8, 32
EPISODES_PER_PRACTICE_SESSION = 5
IMAGE_SHAPE = (128, 128, 4)


class Agent:
    """Entity which learns to solve tasks using bricks"""

    def __init__(self, tasks):
        self.tasks = tasks
        self.parameters = AttrDict({})
        code_atoms = self.pull_numbers("code_atoms",
                                       MIN_CODE_ATOMS, MAX_CODE_ATOMS)
        code_channels = self.pull_numbers("code_channels",
                                          MIN_CODE_CHANNELS, MAX_CODE_CHANNELS)
        self.code_spec = get_spec(shape=(code_atoms, code_channels),
                                  format="code")
        self.code_spec.size = get_size(self.code_spec.shape)
        # we add a sensor for task id
        n_tasks = len(self.tasks.keys())
        task_id_spec = get_spec(shape=n_tasks, format="onehot")
        self.task_sensor = Interface(self, "task_key",
                                     task_id_spec, self.code_spec)
        # we add a sensor for images
        self.image_spec = get_spec(shape=IMAGE_SHAPE, format="image",
                                   add_coords=True)
        self.image_sensor = Interface(self, "image_sensor",
                                      self.image_spec, self.code_spec)
        self.image_actuator = Interface(self, "image_actuator",
                                        self.code_spec, self.image_spec)
        # we add task sensors
        self.tasks = map_attrdict(self.add_sensors_and_actuators, tasks)
        self.decide_n_in_n_out()
        self.shared_model = GraphModel(self)
        self.replay = []

    def decide_n_in_n_out(self):
        self.n_in = max([len(task_dict.inputs)
                         for task_id, task_dict in self.tasks.items()])
        self.n_out = max([len(task_dict.outputs)
                          for task_id, task_dict in self.tasks.items()])

    def add_sensors_and_actuators(self, task_key, task_dict):
        """Add interfaces to a task_dict"""
        sensors, actuators = [], []
        for input_number, in_spec in enumerate(task_dict.inputs):
            if in_spec.format is "image":
                self.image_sensor.add_channel_changer(in_spec.shape[-1])
                sensor = self.image_sensor
                actuator = self.image_actuator
            else:
                sensor = Interface(self, task_key, in_spec, self.code_spec,
                                   input_number=input_number)
                actuator = Interface(self, task_key, self.code_spec, in_spec,
                                     input_number=input_number)
            sensors.append(sensor)
            actuators.append(actuator)
        for output_number, out_spec in enumerate(task_dict.outputs):
            if out_spec.format is "image":
                actuator = self.image_actuator
            else:
                actuator = Interface(self, task_key, self.code_spec, out_spec)
            actuators.append(actuator)
        task_dict.actuators = list(actuators)
        task_dict.sensor = list(sensors)
        return task_key, task_dict

    def train(self):
        """Run EPISODES_PER_PRACTICE_SESSION episodes"""
        [[v.task_runner(self, k, v)
          for k, v in self.tasks.items()]
            for episode_number in range(EPISODES_PER_PRACTICE_SESSION)]

    def __call__(self, task_id, task_dict, inputs):
        """
        Encode, reconstruct, predict, and decide for a task's input[s]

        Args:
            task_id: string key of the task
            task_dict: dictionary for the task w/ sensors
            inputs: a list of tensors

        Returns: tuple of tensors
            normies, codes, reconstructions, predictions, actions
        """
        for output_number, output in enumerate(task_dict.outputs):
            if "variables" in output.keys():
                for var_id, var_position in output.variables:
                    value = inputs[output_number][var_position]
                    self.parameters[var_id] = value
        self.current_task_dict = task_dict
        task_input = get_onehot(task_id, self.tasks.keys())
        task_code = self.task_sensor(task_input)
        codes = map_enumerate(task_dict.sensors, inputs)
        codes = [task_code] + codes
        judgments = self.shared_model(codes)
        world_model = tf.concat([*codes, *judgments], 0)
        actions = map_enumerate(task_dict.actuators, world_model, task_dict)
        return codes, judgments, actions

    def pull_numbers(self, pkey, a, b, step=1, n=1):
        """
        WARNING!: IF KEY EXISTS, RETURNS CURRENT VALUE, NOT A NEW ONE!

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
        elif isinstance(a, int) and isinstance(b, int) and a != b:
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
        WARNING!: IF KEY EXISTS, RETURNS CURRENT VALUE, NOT A NEW ONE!

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
            maybe_choice_or_choices = random.sample(options, int(n))
        else:
            maybe_choice_or_choices = np.random.choice(options, n, p=distribution).tolist()
        if n is 1:
            maybe_choice_or_choices = maybe_choice_or_choices[0]
        self.parameters[pkey] = maybe_choice_or_choices
        return maybe_choice_or_choices

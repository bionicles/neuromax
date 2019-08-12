# agent.py - handle multitask AI
from attrdict import AttrDict
import tensorflow as tf
import numpy as np
import random

from tools.map_attrdict import map_attrdict
from tools.sum_entropy import sum_entropy
from tools.show_model import show_model
from tools.get_size import get_size
from tools.get_spec import get_spec

from nature.bricks.graph_model.graph_model import GraphModel
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
        self.code_atoms = self.pull_numbers(
            "code_atoms", MIN_CODE_ATOMS, MAX_CODE_ATOMS)
        self.code_channels = self.pull_numbers(
            "code_channels", MIN_CODE_CHANNELS, MAX_CODE_CHANNELS)
        self.code_spec = get_spec(shape=(self.code_atoms, self.code_channels),
                                  format="code")
        self.code_spec.size = get_size(self.code_spec.shape)
        self.loss_spec = get_spec(format="float", shape=tuple(1))
        # we add a sensor for task id
        n_tasks = len(self.tasks.keys())
        self.task_id_spec = get_spec(shape=n_tasks, format="onehot")
        self.task_sensor = Interface(self, "task_key",
                                     self.task_id_spec, self.code_spec)
        # we add a sensor for images
        self.image_spec = get_spec(shape=IMAGE_SHAPE, format="image",
                                   add_coords=True)
        self.image_sensor = Interface(self, "image_sensor",
                                      self.image_spec, self.code_spec)
        self.image_actuator = Interface(self, "image_actuator",
                                        self.code_spec, self.image_spec)
        # we add task-specific sensors and actuators
        self.tasks = map_attrdict(self.add_sensors_and_actuators, self.tasks)
        # we build the shared GraphModel
        self.decide_n_in_n_out()
        self.shared_model = GraphModel(self)
        # we build a keras model for each task
        self.tasks = map_attrdict(self.make_task_model, self.tasks)

    def decide_n_in_n_out(self):
        self.max_in = max([len(task_dict.inputs)
                           for task_id, task_dict in self.tasks.items()])
        self.max_out = max([len(task_dict.outputs)
                            for task_id, task_dict in self.tasks.items()])

    def add_sensors_and_actuators(self, task_key, task_dict):
        """Add interfaces to a task_dict"""
        sensors, actuators = [], []
        task_dict.loss_sensor = Interface(self, task_key + "loss_sensor",
                                          self.loss_spec, self.code_spec)
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
        """Run EPISODES_PER_PRACTICE_SESSION episodes
        uses functions stored in self.tasks (indicated in neuromax.py tasks)
        """
        [[task_dict.run_agent_on_task(self, task_key, task_dict)
          for task_key, task_dict in self.tasks.items()]
            for episode_number in range(EPISODES_PER_PRACTICE_SESSION)]

    def make_task_model(self, task_id, task_dict):
        # we make an input for the task_id and encode it
        task_id_input = K.Input(self.task_id_spec.shape)
        task_code = self.task_sensor(task_id_input)
        # likewise for loss float value
        loss_input = K.Input((1))
        loss_code = self.loss_sensor(loss_input)
        # we track lists of all the things
        inputs = [task_id_input, loss_input]
        codes = [task_code, loss_code]
        normies = []
        # now we'll encode the inputs
        for i in range(len(task_dict.inputs)):
            # we make an input
            input = K.Input(task_dict.inputs[i].shape)
            # we use it on the sensors to get normies and codes
            normie, input_code = task_dict.sensors[i](input)
            inputs = inputs + [input]
            normies = normies + [normie]
            codes = codes + [input_code]
        # the full code is task_code, loss_code, input_codes
        code = tf.concat(codes, 0)
        # we pass the codes to the shared model to get judgments
        judgment = self.shared_model(code)
        world_model = tf.concat([code, judgment], 0)
        # we predict next code
        code_prediction = self.code_predictor(world_model)
        world_model = tf.concat([world_model, code_prediction], 0)
        loss_prediction = self.loss_predictor(world_model)
        actions = [code_prediction, loss_prediction]
        # we pass codes and judgments to actuators to get actions
        for o in range(len(task_dict.outputs)):
            action = task_dict.actuators[o](world_model)
            actions = actions + [action]
        # we build a model
        outputs = normies + code + judgment + actions
        task_model = K.Model(inputs, outputs, name=f"{task_id}_model")
        show_model(task_model, ".", task_id, ".png")
        task_dict.model = task_model

    def compute_code_shape(self, task_dict):
        n_in = len(task_dict.inputs)
        n_out = len(task_dict.outputs)
        # task_code, loss_code, input_codes (n_in), prior_output_codes (n_out)
        return (self.code_atoms * (2 + n_in + n_out), self.code_channels)

    def unpack_actions(task_dict, actions):
        code_prediction = actions[0]
        loss_prediction = actions[1]
        n_in = len(task_dict.inputs)
        reconstructions = actions[2:2 + n_in]
        outputs = actions[3 + n_in:]
        if len(outputs) == 1:
            outputs = outputs[0]
        return code_prediction, loss_prediction, reconstructions, outputs

    def compute_free_energy(
        loss=None, prior_loss_prediction=None,
        normies=None, reconstructions=None,
        code=None, prior_code_prediction=None,
        actions=None
    ):
        loss_surprise = -1 * prior_loss_prediction.log_prob(loss)
        code_surprise = -1 * prior_code_prediction.log_prob(code)
        reconstruction_errors = [
            tf.keras.losses.MSE(normie, reconstruction)
            for normie, reconstruction in zip(normies, reconstructions)]
        reconstruction_error = tf.math.reduce_sum(reconstruction_errors)
        freedom = sum_entropy(actions)
        free_energy = loss_surprise + code_surprise + reconstruction_error - freedom
        return free_energy

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

    def pull_tensor(self, pkey, shape, method=tf.random.normal,
                    dtype=tf.float32):
        """
        WARNING!: IF KEY EXISTS, RETURNS CURRENT VALUE, NOT A NEW ONE!

        Pull a tensor from the agent.
        Random normal sampled float32 unless method and dtype are specified

        Args:
            pkey: unique key to share / not share values
            shape: desired tensor shape
            method: the function to call (default tf.random.normal)

        Returns:
            sampled_tensor: tensor of shape
        """
        if pkey in self.parameters.keys():
            return self.parameters[pkey]
        sampled_tensor = method(shape, dtype=dtype)
        self.parameters[pkey] = sampled_tensor
        return sampled_tensor

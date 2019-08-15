# agent.py - handle multitask AI
import tensorflow_probability as tfp
from attrdict import AttrDict
import tensorflow as tf
import numpy as np
import random

from tools.map_attrdict import map_attrdict
from tools.sum_entropy import sum_entropy
from tools.show_model import show_model
from tools.normalize import normalize
from tools.get_spec import get_spec
from tools.get_size import get_size
from tools.log import log

from nature.bricks.graph_model.graph_model import GraphModel
from nature.bricks.interface import Interface
from nature.bricks.get_mlp import get_mlp

tfd = tfp.distributions
K = tf.keras
L = K.layers

MIN_CODE_ATOMS, MAX_CODE_ATOMS = 8, 32
EPISODES_PER_PRACTICE_SESSION = 5
IMAGE_SHAPE = (64, 64, 4)
BATCH_SIZE = 1


class Agent:
    """Entity which learns to solve tasks using bricks"""

    def __init__(self, tasks):
        self.batch_size = BATCH_SIZE
        self.tasks = tasks
        self.parameters = AttrDict({})
        self.code_atoms = self.pull_numbers(
            "code_atoms", MIN_CODE_ATOMS, MAX_CODE_ATOMS)
        self.code_spec = get_spec(shape=(self.code_atoms, 1), format="code")
        self.code_spec.size = self.code_atoms
        self.loss_spec = get_spec(format="float", shape=(1,))
        # we add a sensor for task id
        n_tasks = len(self.tasks.keys())
        self.task_id_spec = get_spec(shape=n_tasks, format="onehot")
        self.task_sensor = Interface(self, "task_id",
                                     self.task_id_spec, self.code_spec)
        # we add a sensor for images
        self.image_spec = get_spec(shape=IMAGE_SHAPE, format="image",
                                   add_coords=True)
        self.image_sensor = Interface(self, "image_sensor",
                                      self.image_spec, self.code_spec)
        self.image_actuator = Interface(self, "image_actuator",
                                        self.code_spec, self.image_spec)
        # we build the shared GraphModel
        self.decide_n_in_n_out()
        self.shared_model = GraphModel(self)
        # we build a keras model for each task
        self.tasks = map_attrdict(self.make_task_model, self.tasks)

    def train(self):
        """Run EPISODES_PER_PRACTICE_SESSION episodes
        uses functions stored in self.tasks (indicated in neuromax.py tasks)
        """
        [[task_dict.run_agent_on_task(self, task_key, task_dict)
          for task_key, task_dict in self.tasks.items()]
            for episode_number in range(EPISODES_PER_PRACTICE_SESSION)]

    def make_task_model(self, task_id, task_dict):
        # we track lists of all the things
        sensors, actuators, normies, codes = [], [], [], []
        # we make an input for the task_id and encode it
        task_id_input = K.Input(self.task_id_spec.shape, batch_size=BATCH_SIZE)
        task_code = self.task_sensor(task_id_input)
        codes.append(task_code)
        # likewise for loss float value
        loss_input = K.Input((1,), batch_size=BATCH_SIZE)
        task_dict.loss_sensor = Interface(self, task_id + "loss_sensor",
                                          self.loss_spec, self.code_spec)
        loss_code = task_dict.loss_sensor(loss_input)
        codes.append(loss_code)
        inputs = [task_id_input, loss_input]
        # now we'll encode the inputs
        for input_number, in_spec in enumerate(task_dict.inputs):
            if in_spec.format is "image":
                sensor = self.image_sensor
                actuator = self.image_actuator
            else:
                sensor = Interface(self, task_id, in_spec, self.code_spec,
                                   input_number=input_number)
                actuator = Interface(self, task_id, self.code_spec, in_spec,
                                     input_number=input_number)
            sensors.append(sensor)
            actuators.append(actuator)
            # we make an input and use it on the sensor to get normies & codes
            input = K.Input(task_dict.inputs[input_number].shape,
                            batch_size=BATCH_SIZE)
            normie, input_code = sensor(input)
            inputs.append(input)
            normies.append(normie)
            codes.append(input_code)
        # the full code is task_code, loss_code, input_codes
        samples = [c.sample(1) for c in codes]
        log("samples", samples, color="green")
        code = tf.concat(samples, -1)
        log("concat code", code, color="green")
        code = tf.einsum('bij->bji', code)
        log("code", code, color="green")
        judgment = self.shared_model(code)
        log("judgment", judgment, color="green")
        world_model = tf.concat([code, judgment], 1)
        log("world_model", world_model, color="green")
        # we predict next code
        task_dict.code_predictor = get_mlp(
            world_model.shape, [(32, "tanh"), (32, "tanh"), (1, "tanh")])
        code_prediction = task_dict.code_predictor(world_model)
        log("code_prediction", code_prediction, color="green")
        world_model = tf.concat([world_model, code_prediction], 1)
        log("world_model w/ code prediction", world_model, color="green")
        flat_world = tf.squeeze(world_model, -1)
        task_dict.loss_predictor = get_mlp(
            flat_world.shape, [(32, "tanh"), (32, "tanh"), (1, "linear")])
        loss_prediction = task_dict.loss_predictor(flat_world)
        log("loss_prediction", loss_prediction, color="green")
        loss_prediction = tf.expand_dims(loss_prediction, -1)
        world_model = tf.concat([world_model, loss_prediction], 1)
        log("world_model w/ loss prediction", world_model, color="green")
        actions = [code_prediction, loss_prediction]
        world_model_spec = get_spec(format="code", shape=world_model.shape[1:])
        log("task world model spec", world_model_spec, color="red")
        # we pass codes and judgments to actuators to get actions
        for output_number, out_spec in enumerate(task_dict.outputs):
            if out_spec.format is "image":
                actuator = self.image_actuator
            else:
                actuator = Interface(self, task_id, world_model_spec, out_spec)
            if out_spec.format is "ragged":
                id, n, index = out_spec.variables[0]
                placeholder = tf.ones_like(inputs[n + 2])
                placeholder = tf.slice(placeholder, [0, 0, 0], [-1, -1, 1])
                action = actuator([placeholder, world_model])
            else:
                action = actuator(world_model)
            log("out_spec", output_number, out_spec, action, color="red")
            actuators.append(actuator)
            actions.append(action)
        task_dict.actuators = list(actuators)
        task_dict.sensor = list(sensors)
        # we build a model
        outputs = [*normies, code, world_model, *actions]
        task_model = K.Model(inputs, outputs, name=f"{task_id}_model")
        show_model(task_model, ".", task_id, "png")
        task_dict.model = task_model
        return task_id, task_dict

    def compute_code_shape(self, task_dict):
        n_in = len(task_dict.inputs)
        n_out = len(task_dict.outputs)
        return (self.code_atoms * (2 + n_in + n_out))

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

    def decide_n_in_n_out(self):
        self.max_in = max([len(task_dict.inputs)
                           for task_id, task_dict in self.tasks.items()])
        self.max_out = max([len(task_dict.outputs)
                            for task_id, task_dict in self.tasks.items()])

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

# agent.py - handle multitask AI
import tensorflow_probability as tfp
from attrdict import AttrDict
import tensorflow as tf
import numpy as np
import random

from tools.map_attrdict import map_attrdict
from tools.show_model import show_model
from tools.get_prior import get_prior
from tools.get_spec import get_spec
from tools.log import log

from nature.bricks.graph_model.graph_model import GraphModel
from nature.bricks.interface import Interface

tfd = tfp.distributions
tfpl = tfp.layers
K = tf.keras
L = K.layers
B = tf.keras.backend

MIN_CODE_ATOMS, MAX_CODE_ATOMS = 4, 16
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
        self.graph_model = GraphModel(self)
        # we build a keras model for each task
        self.tasks = map_attrdict(self.pull_model, self.tasks)

    def train(self):
        """Run EPISODES_PER_PRACTICE_SESSION episodes
        uses functions stored in self.tasks (indicated in neuromax.py tasks)
        """
        [[task_dict.run_agent_on_task(self, task_key, task_dict)
          for task_key, task_dict in self.tasks.items()]
            for episode_number in range(EPISODES_PER_PRACTICE_SESSION)]

    def compute_free_energy(
        self, loss, outputs, task_dict, prior_predictions
    ):
        normies, reconstructions, codes, predictions, actions = self.unpack(
            task_dict.output_roles, outputs)
        surprise = tf.math.reduce_sum([
            -1 * prediction.log_prob(code)
            for prediction, code in zip(prior_predictions, codes)])
        reconstruction_error = 0.
        for normie, reconstruction in zip(normies, reconstructions):
            if hasattr(reconstruction, "entropy"):
                error = -1 * reconstruction.log_prob(normie)
                error = error - reconstruction.entropy()
            else:
                error = tf.keras.losses.MSE(normie, reconstruction)
            reconstruction_error = reconstruction_error + error
        freedom = tf.math.reduce_sum([action.entropy() for action in actions])
        free_energy = loss + reconstruction_error + surprise - freedom
        return free_energy, predictions

    def decide_n_in_n_out(self):
        """
        Decide how many inputs and outputs are needed for the graph_model
        n_in = task_code + loss_code + max(len(task_dict.inputs))
        n_out = n_in predictions + max(len(task_dict.outputs))
        """
        self.n_in = 2 + max([len(task_dict.inputs)
                             for _, task_dict in self.tasks.items()])
        self.n_out = self.n_in + max([len(task_dict.outputs)
                                      for _, task_dict in self.tasks.items()])

    def pull_model(self, task_id, task_dict):
        print("")
        log("MAKE_TASK_MODEL FOR {task_id}",
            color="black_on_white")
        # we track lists of all the things
        codes, outputs, output_roles = [], [], []
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
            # we'll need a sensor and an actuator
            if in_spec.format is "image":
                sensor = self.image_sensor
                actuator = self.image_actuator
            else:
                sensor = Interface(self, f"{task_id}_sensor",
                                   in_spec, self.code_spec,
                                   input_number=input_number)
                actuator = Interface(self, f"{task_id}_actuator",
                                     self.code_spec, in_spec,
                                     input_number=input_number)
            # we make an input and use it on the sensor to get normies & codes
            input = K.Input(task_dict.inputs[input_number].shape,
                            batch_size=BATCH_SIZE)
            normie, input_code = sensor(input)
            outputs.append(normie)
            output_roles.append("normie")
            outputs.append(input_code)
            output_roles.append(f"code-{input_number}")
            codes.append(input_code)
            inputs.append(input)
            # now we reconstruct the normie
            if in_spec.format is "ragged":
                placeholder = tf.ones_like(normie)
                placeholder = tf.slice(placeholder, [0, 0, 0], [-1, -1, 1])
                reconstruction = actuator([input_code, placeholder])
            else:
                reconstruction = actuator(input_code)
            outputs.append(reconstruction)
            output_roles.append(f"reconstruction-{input_number}")
        # graph_model always expects the max number of codes
        # so we make placeholders
        n_placeholders = self.n_in - (2 + len(task_dict.inputs))
        [codes.append(get_prior(self.code_spec.shape)) for _ in range(n_placeholders)]
        judgments = self.graph_model(codes)
        # we save the predictions
        for judgment_number, judgment in enumerate(judgments):
            if judgment_number < (self.n_in - n_placeholders):
                output_roles.append(f"prediction-{judgment_number}")
                outputs.append(judgment)
        judgment = tf.concat([j.sample() for j in judgments], 1)
        log("judgment", judgment.shape, color="yellow")
        code = tf.concat([j.sample() for j in codes], 1)
        log("code", code.shape, color="yellow")
        world_model = tf.concat([judgment, code], 1)
        world_model_spec = get_spec(format="code", shape=world_model.shape)
        log("world_model", world_model, color="green")
        log("world_model_spec", world_model_spec, color="green")
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
                action = actuator([world_model, placeholder])
            else:
                action = actuator(world_model)
            outputs.append(action)
            output_roles.append(f"action-{output_number}")
        # we build a model
        task_model = K.Model(inputs, outputs, name=f"{task_id}_model")
        task_dict.output_roles = output_roles
        task_dict.model = task_model
        show_model(task_model, ".", task_id, "png")
        [log(role, output.shape, color="green") for role, output in zip(output_roles, outputs)]
        return task_id, task_dict

    def compute_code_shape(self, task_dict):
        n_in = len(task_dict.inputs)
        n_out = len(task_dict.outputs)
        return (self.code_atoms * (2 + n_in + n_out))

    @staticmethod
    def unpack(output_roles, outputs):
        normies, reconstructions, actions = [], [], []
        for role, output in zip(output_roles, outputs):
            log("unpack", role, output.shape, color="blue")
            if role == "normie":  # tensor
                normies.append(output)
            elif role == "reconstruction":  # tensor
                reconstructions.append(output)
            elif role == "code_prediction":  # distribution
                code_prediction = output
            elif role == "loss_prediction":  # distribution
                loss_prediction = output
            elif role == "code":  # tensor
                code = output
            elif "action" in role:  # distribution
                actions.append(output)
        return (normies, reconstructions, code, code_prediction,
                loss_prediction, actions)

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
            maybe_choice_or_choices = np.random.choice(
                options, n, p=distribution).tolist()
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

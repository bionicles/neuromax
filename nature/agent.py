# agent.py - handle multitask AI
import tensorflow_probability as tfp
from attrdict import AttrDict
import tensorflow as tf
import numpy as np
import random

from tools.map_attrdict import map_attrdict
from tools.sum_entropy import sum_entropy
from tools.show_model import show_model
from tools.get_spec import get_spec
from tools.get_size import get_size
from tools.log import log

from nature.bricks.graph_model.graph_model import GraphModel
from nature.bricks.interface import Interface

tfd = tfp.distributions
tfpl = tfp.layers
K = tf.keras
L = K.layers
B = tf.keras.backend

UNITS, FN = 32, "tanh"
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
        log("\n\nMAKE_TASK_MODEL FOR {task_id}",
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
                sensor = Interface(self, task_id, in_spec, self.code_spec,
                                   input_number=input_number)
                actuator = Interface(self, task_id, self.code_spec, in_spec,
                                     input_number=input_number)
            # we make an input and use it on the sensor to get normies & codes
            input = K.Input(task_dict.inputs[input_number].shape,
                            batch_size=BATCH_SIZE)
            normie, input_code = sensor(input)
            outputs.append(normie)
            output_roles.append("normie")
            codes.append(input_code)
            inputs.append(input)
            if in_spec.format is "ragged":
                placeholder = tf.ones_like(normie)
                placeholder = tf.slice(placeholder, [0, 0, 0], [-1, -1, 1])
                reconstruction = actuator([input_code, placeholder])
            else:
                reconstruction = actuator(input_code)
            outputs.append(reconstruction)
            output_roles.append("reconstruction")
        # the full code is task_code, loss_code, input_codes
        samples = [c.sample() for c in codes]
        [log(sample.shape, color="yellow") for sample in samples]
        code = tf.einsum('ijk->kij', tf.concat(samples, -1))
        code = tf.reshape(code, (1, -1, 1))
        log("code", code, color="yellow")
        outputs.append(code)
        output_roles.append("code")
        judgment = self.shared_model(code)
        world_model = tf.concat([code, judgment], 1)
        # we predict next code
        # squeezed_model = tf.squeeze(world_model, 0)
        # log("squeezed_model", squeezed_model, color="red")
        code_predictor = self.pull_distribution(world_model.shape, code.shape)
        code_prediction = code_predictor(world_model)
        log("code_prediction", code_prediction, color="red_on_white")
        outputs.append(code_prediction)
        output_roles.append("code_prediction")
        # now we add the code prediction to the world model and predict loss
        code_prediction_sample = code_prediction.sample()
        log("code_prediction_sample", code_prediction_sample.shape, color="red_on_white")
        try:
            world_model = B.concatenate([world_model, code_prediction_sample], -1)
        except Exception as e:
            log(e, color="white_on_red")
            world_model, code_prediction_sample = self.fix_shapes(world_model), self.fix_shapes(code_prediction_sample)
            world_model = B.concatenate([world_model, code_prediction_sample], -1)
        log("world_model", world_model, color="yellow")
        loss_predictor = self.pull_distribution(
            world_model.shape, self.loss_spec.shape)
        loss_prediction = loss_predictor(world_model)
        outputs.append(loss_prediction)
        output_roles.append("loss_prediction")
        world_model = B.concatenate([world_model, loss_prediction], -1)
        world_model_spec = get_spec(format="code", shape=world_model.shape[1:])
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

    def make_normal(self, output, role):
        should_be_distribution = "prediction" in role or "action" in role
        not_a_distribution = not hasattr(output, "entropy")
        if should_be_distribution and not_a_distribution:
            log("make a distribution for", role)
            loc, scale = tf.split(output, 2, axis=-1)
            return tfd.Normal(loc=loc, scale=scale)
        return output

    def pull_distribution(self, in_shape, desired_shape, batch_size=1,
                          units=UNITS, fn=FN):
        in_shape = list(in_shape)[1:]
        shapes = tfd.Normal.param_shapes(desired_shape)
        loc, scale = tf.zeros(shapes["loc"]), tf.ones(shapes["scale"])
        prior_tensor = tf.concat([loc, scale], -1)
        prior_tensor = tf.expand_dims(prior_tensor, -1)
        last_layer_units = get_size(prior_tensor.shape)
        prior = tfd.Normal(loc, scale)
        log(f"pull_distribution {in_shape} --> {desired_shape}", color="red")
        log(f"pull_distribution prior_tensor.shape {prior_tensor.shape}", color="red")
        log(f"pull_distribution prior {prior}", color="red")
        log(f"pull_distribution prior_sample {prior.sample().shape}", color="red")
        return K.Sequential([
            K.Input(in_shape, batch_size=batch_size),
            L.Flatten(),
            tfpl.DenseReparameterization(units, fn),
            tfpl.DenseReparameterization(units, fn),
            tfpl.DenseReparameterization(last_layer_units),
            L.Reshape(prior_tensor.shape),
            tfpl.DistributionLambda(
                make_distribution_fn=lambda x: tfd.Normal(
                    loc=x[..., 0, :], scale=tf.math.abs(x[..., 1, :])
                ),
                convert_to_tensor_fn=lambda s: s.sample(),
                activity_regularizer=tfpl.KLDivergenceRegularizer(prior)
            )
        ])

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

    def compute_free_energy(
        self, loss, outputs, task_dict,
        prior_code_prediction, prior_loss_prediction
    ):
        normies, reconstructions, code, code_prediction, loss_prediction, actions = self.unpack(task_dict.output_roles, outputs)
        loss_surprise = -1 * prior_loss_prediction.log_prob(loss)
        code_surprise = -1 * prior_code_prediction.log_prob(code)
        reconstruction_errors = [
            tf.keras.losses.MSE(normie, reconstruction)
            for normie, reconstruction in zip(normies, reconstructions)]
        reconstruction_error = tf.math.reduce_sum(reconstruction_errors)
        freedom = sum_entropy(actions)
        free_energy = loss + loss_surprise + code_surprise + reconstruction_error - freedom
        return free_energy, code_prediction, loss_prediction

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

    def fix_shapes(self, tensor):
        axis = 0
        for axis_shape in tensor.shape[1:]:
            axis += 1
            if axis_shape == 1:
                tensor = tf.squeeze(tensor, axis=axis)
                axis -= 1
        return tensor

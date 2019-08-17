# agent.py - handle multitask AI
import tensorflow_probability as tfp
from attrdict import AttrDict
import tensorflow as tf
import numpy as np
import random
import time

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
        log("we add a task id sensor", color="green")
        n_tasks = len(self.tasks.keys())
        self.task_id_spec = get_spec(shape=n_tasks, format="onehot")
        self.task_sensor = Interface(self, "task_id",
                                     self.task_id_spec, self.code_spec)
        log("we add a sensor for images", color="green")
        self.image_spec = get_spec(shape=IMAGE_SHAPE, format="image",
                                   add_coords=True)
        self.image_sensor = Interface(self, "image_sensor",
                                      self.image_spec, self.code_spec)
        self.image_actuator = Interface(self, "image_actuator",
                                        self.code_spec, self.image_spec)
        log("we build the shared GraphModel", color="green")
        self.decide_n_in_n_out()
        self.graph_model = GraphModel(self)
        log("we build a keras model for each task", color="green")
        self.tasks = map_attrdict(self.pull_model, self.tasks)
        self.priors = [tf.zeros(self.code_spec.shape)
                       for _ in range(self.n_in)]
        self.optimizer = tf.keras.optimizers.Adam(amsgrad=True)

    def pull_model(self, task_id, task_dict):
        print("")
        log(f"agent.pull_model {task_id}", color="black_on_white")
        outputs, output_roles = [], []
        log("we encode the task's id", color="green")
        task_id_input = K.Input(self.task_id_spec.shape, batch_size=BATCH_SIZE)
        task_code = self.task_sensor(task_id_input)
        codes = [task_code]
        log("likewise for loss float value", color="green")
        loss_input = K.Input((1,), batch_size=BATCH_SIZE)
        task_dict.loss_sensor = Interface(self, task_id + "loss_sensor",
                                          self.loss_spec, self.code_spec)
        loss_code = task_dict.loss_sensor(loss_input)
        codes.append(loss_code)
        inputs = [task_id_input, loss_input]
        log(f"we autoencode {task_id} inputs", color="green")
        for in_number, in_spec in enumerate(task_dict.inputs):
            log(f"input {in_number} needs a sensor and an actuator", color="green")
            if in_spec.format is "image":
                sensor = self.image_sensor
                actuator = self.image_actuator
            else:
                sensor = Interface(self, f"{task_id}_sensor",
                                   in_spec, self.code_spec,
                                   in_number=in_number)
                actuator = Interface(self, f"{task_id}_actuator",
                                     self.code_spec, in_spec,
                                     in_number=in_number)
            log("we pass an input to the sensor to get normies & codes",
                color="green")
            input = K.Input(task_dict.inputs[in_number].shape,
                            batch_size=BATCH_SIZE)
            normie, input_code = sensor(input)
            outputs.append(normie)
            output_roles.append(f"normie-{in_number}")
            outputs.append(input_code)
            output_roles.append(f"code-{in_number}")
            codes.append(input_code)
            inputs.append(input)
            log("now we reconstruct the normie from the code", color="green")
            if in_spec.format is "ragged":
                placeholder = tf.ones_like(normie)
                placeholder = tf.slice(placeholder, [0, 0, 0], [-1, -1, 1])
                reconstruction = actuator([input_code, placeholder])
            else:
                reconstruction = actuator(input_code)
            outputs.append(reconstruction)
            output_roles.append(f"reconstruction-{in_number}")
        log("we make placeholders for agent.graph_model", color="green")
        n_placeholders = self.n_in - (2 + len(task_dict.inputs))
        if n_placeholders < 1:
            log("no placeholders to make...moving on", color="green")
        else:
            for _ in range(n_placeholders):
                prior, _ = get_prior(self.code_spec.shape)
                codes.append(prior)
        log(f"GraphModel prefers tensors, so we sample {len(codes)} codes", color="green")
        samples = [c.sample() for c in codes]
        samples = [tf.expand_dims(s, 0) if len(s.shape) < 3 else s
                   for s in samples]
        log("now we pass code samples to GraphModel:", color="green")
        predictions = self.graph_model(samples)
        log("then we save the predictions", color="green")
        for prediction_number, prediction in enumerate(predictions):
            distribution = Interface(
                self, "shared", get_spec(
                    format="code", shape=prediction.shape),
                self.code_spec)(prediction)
            output_roles.append(f"prediction-{prediction_number}")
            outputs.append(distribution)
        log("we assemble a world model for the actuators", color="green")
        predictions = [
            tf.expand_dims(p, 0) if len(p.shape) < 3 else p
            for p in predictions]
        world_model = tf.concat([*samples, *predictions], 1)
        world_model_spec = get_spec(format="code", shape=world_model.shape)
        log("we pass codes and judgments to actuators to get actions",
            color="green")
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
        log("")
        log("we build a model with inputs:", color="green")
        [log("input", n, list(i.shape), color="yellow") for n, i in enumerate(inputs)]
        log("")
        log("and outputs:", color="green")
        self.unpack(output_roles, outputs)
        task_model = K.Model(inputs, outputs, name=f"{task_id}_model")
        task_dict.output_roles = output_roles
        task_dict.model = task_model
        show_model(task_model, ".", task_id, "png")
        log("")
        log(f"SUCCESS! WE BUILT A {task_id.upper()} MODEL!", color="green")
        return task_id, task_dict

    @staticmethod
    def unpack(output_roles, outputs):
        normies, codes, reconstructions, predictions, actions = [], [], [], [], []
        for role, output in zip(output_roles, outputs):
            log("unpack", role, list(output.shape), color="yellow")
            if "normie" in role:  # tensor
                normies.append(output)
            elif "code" in role:  # tensor
                codes.append(output)
            elif "reconstruction" in role:  # tensor
                reconstructions.append(output)
            elif "prediction" in role:  # distribution
                predictions.append(output)
            elif "action" in role:  # distribution
                actions.append(output)
        return (normies, reconstructions, codes, predictions, actions)

    @staticmethod
    def compute_error_or_surprise(y_true_list, y_pred_list):
        total = 0.
        for n, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
            log("")
            log("pair", n)
            entropy = 0.
            log("y_true", y_true.shape, color="red")
            log("y_pred", y_pred.shape, color="white")
            log("")
            time.sleep(1)
            if hasattr(y_pred, "entropy"):
                if hasattr(y_true, "entropy"):
                    log("y_true and y_pred are both distributions")
                    error = tfd.kl_divergence(y_true, y_pred)
                    string = "kl_divergence(y_true, y_pred)"
                else:
                    log("y_true is a tensor and y_pred is a distribution")
                    error = -1 * y_pred.log_prob(y_true)
                    string = "negative log-probability of y_true given y_pred"
                entropy = tf.reduce_sum(y_pred.entropy())
            else:
                if hasattr(y_true, "entropy"):
                    log("y_true is a distribution and y_pred is a tensor")
                    error = -1 * y_true.log_prob(y_pred)
                    string = "negative log-probability of y_pred given y_true"
                else:
                    log("y_true and y_pred are both tensors")
                    error = tf.keras.losses.MSE(y_true, y_pred)
                    string = "MSE loss"
            error = tf.reduce_sum(error)
            log(error.numpy().tolist() if not isinstance(error, float) else error, string)
            log(entropy.numpy().tolist() if not isinstance(entropy, float) else entropy, "y_pred entropy")
            error = error - entropy
            log(error.numpy().tolist() if not isinstance(error, float) else error, "error - entropy")
            total = total + error
        log(total.numpy().tolist() if not isinstance(total, float) else total, "total")
        return total

    def compute_free_energy(
        self, loss, outputs, prior_predictions, task_dict
    ):
        log("free_energy = loss + error + surprise - freedom", color="green")
        normies, reconstructions, codes, predictions, actions = self.unpack(
            task_dict.output_roles, outputs)
        error_or_surprise = loss
        log("")
        log("we compute reconstruction error/surprise", color="green")
        # log("normies", normies)
        # log("reconstructions", reconstructions)
        error_or_surprise = error_or_surprise + self.compute_error_or_surprise(
            normies, reconstructions)
        log("we compute prediction error/surprise", color="green")
        # log("codes", codes)
        # log("predictions", predictions)
        error_or_surprise = error_or_surprise + self.compute_error_or_surprise(
            codes, prior_predictions
        )
        freedom = tf.math.reduce_sum([action.entropy() for action in actions])
        free_energy = loss + error_or_surprise - freedom
        log("free energy", free_energy, color="green")
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

    def train(self):
        """Run EPISODES_PER_PRACTICE_SESSION episodes
        uses functions stored in self.tasks (indicated in neuromax.py tasks)
        """
        [[task_dict.run_agent_on_task(self, task_key, task_dict)
          for task_key, task_dict in self.tasks.items()]
            for episode_number in range(EPISODES_PER_PRACTICE_SESSION)]

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

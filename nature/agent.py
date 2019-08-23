# agent.py - handle multitask AI
from attrdict import AttrDict
import tensorflow as tf
import numpy as np
import random

from tools import get_percent, get_spec, add_up, log
from nature import use_graph_model
# from nature import Brick

SWITCH_TO_MAE_LOSS_WHEN_FREE_ENERGY_BELOW = 4.2
OPTIMIZER = tf.keras.optimizers.SGD(3e-4, clipvalue=0.04)
EPISODES_PER_PRACTICE_SESSION = 100
LOSS_FN = tf.keras.losses.MSLE

IMAGE_SHAPE = (32, 32, 4)
CODE_CHANNELS = 1
CODE_ATOMS = 32
BATCH_SIZE = 1

K = tf.keras
B = K.backend
L = K.layers


class Agent:
    """Entity which learns to solve tasks using bricks"""

    def __init__(self, tasks):
        self.batch_size = BATCH_SIZE
        self.tasks = tasks
        self.bricks = AttrDict({})
        self.code_spec = get_spec(
            shape=(CODE_ATOMS, CODE_CHANNELS), format="code")
        n_tasks = len(self.tasks.keys())
        self.task_id_spec = get_spec(shape=n_tasks, format="onehot")
        self.image_spec = get_spec(shape=IMAGE_SHAPE, format="image")
        self.loss_spec = get_spec(format="float", shape=(1,))
        self.decide_n_in_n_out()
        self.optimizer = OPTIMIZER
        self.n_gradient_steps = 0

    def pull_model(self, inputs, outputs):
        log("pull_model", inputs, outputs)
        in_specs = [get_spec(input) for input in inputs]
        out_specs = [get_spec(output) for output in outputs]
        return use_graph_model(inputs, in_specs, outputs, out_specs)

    @staticmethod
    def unpack(output_roles, outputs):
        normies, codes, reconstructions, predictions, actions = [], [], [], [], []
        for n, (role, output) in enumerate(zip(output_roles, outputs)):
            log(n, role, list(output.shape), color="yellow")
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

    def compute_errors(self, y_true_list, y_pred_list):
        errors = []
        for n, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
            log("")
            log("pair", n,
                "y_true", list(y_true.shape), "y_pred", list(y_pred.shape))
            error = self.loss_fn(y_true, y_pred)
            errors.append(error)
        return errors

    def train_op(self, task_id, inputs, action_index, y_true, loss_fn, priors):
        log("free_energy = loss + error + surprise - freedom", color="green")
        task_dict = self.tasks[task_id]
        model = task_dict.model
        with tf.GradientTape() as tape:
            losses = []
            outputs = model(inputs)
            normies, reconstructions, codes, predictions, actions = self.unpack(
                task_dict.output_roles, outputs)
            log("we get the action", color="green")
            y_pred = actions[action_index]
            log("we approximate action conformity", color="green")
            if task_dict.outputs[action_index].format is "onehot":
                if task_dict.histogram is None:
                    task_dict.histogram = y_pred
                else:
                    task_dict.histogram = task_dict.histogram + y_pred
                total_actions = tf.math.reduce_sum(task_dict.histogram)
                normalized = task_dict.histogram / total_actions
                equal_prior = tf.ones_like(y_true)
                equal_prior = equal_prior / tf.math.reduce_sum(equal_prior)
                conformity = K.losses.KLD(equal_prior, normalized)
                losses.append(conformity)
            log("we compute task_loss", color="green")
            loss = loss_fn(y_true, y_pred)
            losses.append(loss)
            log("we compute reconstruction error/surprise", color="green")
            reconstruction_errors = self.compute_errors(
                normies, reconstructions)
            [losses.append(e) for e in reconstruction_errors]
            log("we compute prediction error/surprise", color="green")
            prediction_errors = self.compute_errors(codes, priors)
            [losses.append(e) for e in prediction_errors]
        gradients = tape.gradient(losses, model.trainable_variables)
        gradients_and_variables = zip(gradients, model.trainable_variables)
        log(len(gradients), "gradients",
            len(model.trainable_variables), "trainable_variables", color="red")
        self.optimizer.apply_gradients(gradients_and_variables)
        self.n_gradient_steps += 1
        n_gradients = sum([1 if v is not None else 0
                           for v in gradients])
        n_variables = len(model.trainable_variables)
        free_energy = add_up(losses)
        total_reconstruction_error = add_up(reconstruction_errors)
        total_prediction_error = add_up(prediction_errors)
        total_conformity = add_up(conformity)
        log("", debug=True)
        log(f"   step: {self.n_gradient_steps} --- task: {task_id}", debug=True)
        log(f"   {get_percent(n_variables, n_gradients)} of variables have gradients", debug=True)
        log(f"   {get_percent(free_energy, total_reconstruction_error)} reconstruction error at {round(total_reconstruction_error, 2)}", debug=True)
        log(f"   {get_percent(free_energy, total_prediction_error)} prediction error at {round(total_prediction_error, 2)}", debug=True)
        log(f"   {get_percent(free_energy, total_conformity)} conformity at {round(total_conformity, 2)}", debug=True)
        log(f"   {get_percent(free_energy, loss)} task loss at {round(add_up(loss), 2)}", debug=True)
        log(f"   {round(free_energy, 1)} free energy", debug=True)
        if free_energy < SWITCH_TO_MAE_LOSS_WHEN_FREE_ENERGY_BELOW:
            log("  switched to Mean Absolute Error loss", color="red", debug=True)
            self.loss_fn = tf.losses.MAE
        return free_energy, predictions

    def decide_n_in_n_out(self):
        """
        Decide how many inputs and outputs are needed for the graph_model
        n_in = task_code + loss_code + max(len(task_dict.inputs))
        n_out = n_in predictions + max(len(task_dict.outputs))
        """
        self.n_in = 2 + max([len(task_dict.inputs)
                             for _, task_dict in self.tasks.items()])
        self.n_out = self.n_in

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
        if pkey in self.graph.keys():
            return self.graph[pkey]
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
        self.graph[pkey] = maybe_number_or_numbers
        return maybe_number_or_numbers

    def pull_choices(self, pkey, options, n=None, p=None, replace=None):
        """
        WARNING!: IF KEY EXISTS, RETURNS CURRENT VALUE, NOT A NEW ONE!

        Choose from a set of options.
        Uniform sampling unless distribution is given

        Args:
            pkey: unique key to share values
            options: list of possible values # default [0, 1]
            n: number of choices to return
            p: probability to return each option in options

        Returns:
            maybe_choice_or_choices: string if n is 1, else a list
        """
        options = [0, 1] if options is None else options
        if pkey in self.graph.keys():
            return self.graph[pkey]
        maybe_choice_or_choices = np.random.choice(
            options, size=n, p=p, replace=replace).tolist()
        self.graph[pkey] = maybe_choice_or_choices
        return maybe_choice_or_choices

    def pull_tensors(self, pkey, shape, n=None, method=tf.random.normal,
                     dtype=tf.float32):
        """
        WARNING!: IF KEY EXISTS, RETURNS CURRENT VALUE, NOT A NEW ONE!

        Pull a tensor from the agent.
        Random normal sampled float32 unless method and dtype are specified

        Args:
            pkey: unique key to share / not share values
            shape: desired tensor shape
            n: number of tensors to pull (default 1)
            method: the function to call (default tf.random.normal)

        Returns:
            sampled_tensor: tensor of shape
        """
        if pkey in self.graph.keys():
            return self.graph[pkey]
        if n is None:
            parameter = method(shape, dtype=dtype)
        else:
            parameter = [method(shape, dtype=dtype) for _ in range(n)]
        self.graph[pkey] = parameter
        return parameter

    # def pull_brick(self, brick_type, **kwargs):
    #     """construct a keras layer from parts and a function to use them
    #         https://stackoverflow.com/a/45102583/4898464
    #     """
    #     log('pull_brick', brick_type, kwargs, color="red")
    #     if "reuse" in kwargs:
    #         if id in self.graph.keys() and kwargs["reuse"]:
    #             return self.bricks[id]
    #     else:
    #         parts = AttrDict(kwargs, brick_type=brick_type)
    #         parts.id = make_id(parts)
    #         return Brick(self, parts)

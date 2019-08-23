# agent.py - handle multitask AI
# from keras_radam import RAdam
from attrdict import AttrDict
import tensorflow as tf
import random

from tools import get_percent, get_spec, add_up, log
import nature
import nurture

OPTIMIZER = tf.keras.optimizers.SGD(3e-4, clipvalue=0.04)
LOSS_FN = tf.keras.losses.MSLE
EPISODES_PER_SESSION = 100
USE_OTHER_LOSSES = False

IMAGE_SHAPE = (32, 32, 4)

BATCH_SIZE = 1
ATOMS = 32
CHANNELS = 1

K = tf.keras
B, L = K.backend, K.layers


class Agent:
    """Entity which learns to solve tasks using bricks"""

    def __init__(self):
        self.batch_size = BATCH_SIZE
        self.code_spec = get_spec(shape=(ATOMS, CHANNELS), format="code")
        self.image_spec = get_spec(shape=IMAGE_SHAPE, format="image")
        self.loss_spec = get_spec(format="float", shape=(1,))
        self.optimizer = OPTIMIZER
        self.n_gradient_steps = 0
        self.train()

    def train(self):
        task = self.pull_task()
        task = self.pull_model(task)
        for episode_number in range(EPISODES_PER_SESSION):
            self.run_episode()

    def pull_task(self):
        if random.random() < 0.5:
            return nurture.prepare_data()
        return nurture.prepare_env()

    def pull_model(self, task):
        task.model, task.roles = nature.use_graph_model(
            self, in_specs=task.in_specs, out_specs=task.out_specs)
        return task

    def run_episode(self, task):
        observation = task.env.reset()
        if task.is_data:
            observation, y_true = observation
        done = False
        while not done:
            with tf.GradientTape() as tape:
                outputs = task.model(observation)
                normies, reconstructions, y_pred = self.unpack(
                    task.roles, outputs)
                observation, reward, done, _ = task.env.step(y_pred)
                loss = -reward
                losses = [loss]
                if USE_OTHER_LOSSES:
                    reconstruction_errors = self.compute_errors(
                        normies, reconstructions)
                    [losses.append(e) for e in reconstruction_errors]
            gradients = tape.gradient(losses, task.model.trainable_variables)
            gradients_and_variables = zip(gradients, task.model.trainable_variables)
            self.optimizer.apply_gradients(gradients_and_variables)

            log("", debug=True)
            self.n_gradient_steps += 1
            n_gradients = sum([1 if v is not None else 0 for v in gradients])
            n_variables = len(task.model.trainable_variables)
            free_energy = add_up(losses)
            log(f"   step: {self.n_gradient_steps} --- task: {task.key}", debug=True)
            log(f"   {get_percent(n_variables, n_gradients)} of variables have gradients", debug=True)
            if USE_OTHER_LOSSES:
                total_reconstruction_error = add_up(reconstruction_errors)
                log(f"   {get_percent(free_energy, total_reconstruction_error)} reconstruction error at {round(total_reconstruction_error, 2)}", debug=True)
            log(f"   {get_percent(free_energy, loss)} task loss at {round(add_up(loss), 2)}", debug=True)
            log(f"   {round(free_energy, 1)} free energy", debug=True)

    @staticmethod
    def unpack(roles, outputs):
        unpacked = AttrDict()
        for n, (role, output) in enumerate(zip(roles, outputs)):
            if role not in unpacked.keys():
                unpacked[role] = [output]
            else:
                unpacked[role].append(output)
        return (unpacked.normies,
                unpacked.reconstructions,
                unpacked.actions)

    def compute_errors(self, y_true_list, y_pred_list):
        errors = []
        for n, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
            errors.append(self.loss_fn(y_true, y_pred))
        return errors

    # def pull_numbers(self, pkey, a, b, step=1, n=1):
    #     """
    #     Provide numbers from range(a, b, step).
    #     WARNING!: IF KEY EXISTS, RETURNS CURRENT VALUE, NOT A NEW ONE!
    #
    #     Args:
    #         pkey: string key for this parameter
    #         a: low end of the range
    #         b: high end of the range
    #         step: distance between options
    #         n: number of numbers to pull
    #
    #     Returns:
    #         maybe_number_or_numbers: a number if n is 1, a list if n > 1
    #         an int if a and b are ints, else a float
    #     """
    #     if pkey in self.graph.keys():
    #         return self.graph[pkey]
    #     if a == b:
    #         maybe_number_or_numbers = [a for _ in range(n)]
    #     if a > b:
    #         a, b = b, a
    #     elif isinstance(a, int) and isinstance(b, int) and a != b:
    #         maybe_number_or_numbers = [
    #             random.randrange(a, b, step) for _ in range(n)]
    #     else:
    #         if step is 1:
    #             maybe_number_or_numbers = [
    #                 random.uniform(a, b) for _ in range(n)]
    #         else:
    #             maybe_number_or_numbers = np.random.choice(
    #                 np.arange(a, b, step),
    #                 size=n).tolist()
    #     if n is 1:
    #         maybe_number_or_numbers = maybe_number_or_numbers[0]
    #     self.graph[pkey] = maybe_number_or_numbers
    #     return maybe_number_or_numbers
    #
    # def pull_choices(self, pkey, options, n=None, p=None, replace=None):
    #     """
    #     Choose from a set of options.
    #     Uniform sampling unless distribution is given
    #     WARNING!: IF KEY EXISTS, RETURNS CURRENT VALUE, NOT A NEW ONE!
    #
    #     Args:
    #         pkey: unique key to share values
    #         options: list of possible values # default [0, 1]
    #         n: number of choices to return
    #         p: probability to return each option in options
    #
    #     Returns maybe_choice_or_choices: string if n is 1, else a list
    #     """
    #     options = [0, 1] if options is None else options
    #     if pkey in self.graph.keys():
    #         return self.graph[pkey]
    #     maybe_choice_or_choices = np.random.choice(
    #         options, size=n, p=p, replace=replace).tolist()
    #     self.graph[pkey] = maybe_choice_or_choices
    #     return maybe_choice_or_choices
    #
    # def pull_tensors(self, pkey, shape, n=None, method=tf.random.normal,
    #                  dtype=tf.float32):
    #     """
    #     Pull a tensor from the agent.
    #     Random normal sampled float32 unless method and dtype are specified
    #     WARNING!: IF KEY EXISTS, RETURNS CURRENT VALUE, NOT A NEW ONE!
    #
    #     Args:
    #         pkey: unique key to share / not share values
    #         shape: desired tensor shape
    #         n: number of tensors to pull (default 1)
    #         method: the function to call (default tf.random.normal)
    #
    #     Returns sampled_tensor: tensor of shape
    #     """
    #     if pkey in self.graph.keys():
    #         return self.graph[pkey]
    #     if n is None:
    #         parameter = method(shape, dtype=dtype)
    #     else:
    #         parameter = [method(shape, dtype=dtype) for _ in range(n)]
    #     self.graph[pkey] = parameter
    #     return parameter

    # from nature import Brick
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

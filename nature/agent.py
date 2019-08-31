# agent.py - handle multitask AI
from keras_radam.training import RAdamOptimizer
from random import choice
import tensorflow as tf

from tools import get_percent, get_spec, add_up, log
from nurture import prepare_env, prepare_data
from nature import TaskModel


TASK_PREPPERS = [prepare_env, prepare_data]
CLASSIFIER_LOSS = tf.losses.categorical_crossentropy
REGRESSER_LOSS = tf.losses.MSLE
EPISODES_PER_SESSION = 10000
OPTIMIZER = RAdamOptimizer()

IMAGE_SHAPE = (32, 32, 1)

BATCH = 1
ATOMS = 4
CHANNELS = 4

K = tf.keras
B, L = K.backend, K.layers


class Agent:
    """Entity which learns to solve tasks using bricks"""

    def __init__(self):
        self.code_spec = get_spec(shape=(ATOMS, CHANNELS), format="code")
        self.image_spec = get_spec(shape=IMAGE_SHAPE, format="image")
        self.classifier_loss = CLASSIFIER_LOSS
        self.regresser_loss = REGRESSER_LOSS
        self.optimizer = OPTIMIZER
        self.batch_size = BATCH
        self.n_steps = 0
        self.replay = []
        self.train()

    def train(self):
        self.pull_task_and_model()
        for episode_number in range(EPISODES_PER_SESSION):
            with tf.device('/gpu:0'):
                self.run_episode()

    def pull_task_and_model(self):
        self.task = choice(TASK_PREPPERS)(self)
        self.task.model = TaskModel(
            self, in_specs=self.task.in_specs, out_specs=self.task.out_specs)

    def clean_observation(self, observation):
        if observation is None:
            return observation
        if not tf.is_tensor(observation):
            observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        if observation.dtype is not tf.float32:
            observation = tf.cast(observation, tf.float32)
        if observation.shape[0] is not self.batch_size:
            observation = tf.expand_dims(observation, 0)
        return observation

    def run_episode(self):
        observation = self.clean_observation(self.task.env.reset())
        done = False
        while not done:
            with tf.GradientTape() as tape:
                outputs = self.task.model(observation)
                if self.task.out_specs[0].format is "discrete":
                    outputs = int(outputs.numpy().item(0))
                new_observation, reward, done, _ = self.task.env.step(outputs)
                new_observation = self.clean_observation(new_observation)
                loss = -reward
                if not tf.is_tensor(loss):
                    loss = tf.convert_to_tensor(loss)
                losses = [loss] + self.task.model.losses
            gradients = tape.gradient(
                losses, self.task.model.trainable_variables)
            gradients_and_variables = zip(
                gradients, self.task.model.trainable_variables)
            self.optimizer.apply_gradients(gradients_and_variables)

            memory = [observation, outputs, losses, new_observation]
            self.replay.append(memory)
            observation = new_observation

            log("", debug=True)
            self.n_steps += 1
            n_gradients = sum([1 if v is not None else 0 for v in gradients])
            n_variables = len(self.task.model.trainable_variables)
            free_energy = add_up(losses)
            log(f" step: {self.n_steps} --- task: {self.task.key}", debug=True)
            percent = get_percent(n_variables, n_gradients)
            log(f" {percent} of variables have gradients", debug=True)
            try:
                loss = add_up(loss)
            except Exception as e:
                log('exception adding up loss', e)
                loss = loss.numpy().item(0)
            percent = get_percent(free_energy, loss)
            log(f" {percent} task loss at {round(loss, 4)}", debug=True)
            log(f" {round(free_energy, 4)} free energy", debug=True)

    def compute_errors(self, true_list, pred_list):
        return [self.regresser_loss(true, pred)
                for true, pred in zip(true_list, pred_list)]

# agent.py - handle multitask AI
from keras_radam.training import RAdamOptimizer
from random import choice
import tensorflow as tf

from tools import get_percent, get_spec, add_up, log
from nurture import prepare_env, prepare_data
from nature import TaskModel

CLASSIFIER_LOSS = tf.losses.categorical_crossentropy
REGRESSER_LOSS = tf.losses.MSLE
TASK_PREPPERS = [prepare_env, prepare_data]
OPTIMIZER = RAdamOptimizer()
DTYPE = tf.float32

BATCH, ATOMS, CHANNELS = 1, 4, 4
IMAGE_SHAPE = (32, 32, 1)

SESSIONS, EPISODES = 42, 42


class Agent:
    """Entity which solves tasks using models made of bricks"""

    def __init__(self):
        self.code_spec = get_spec(shape=(ATOMS, CHANNELS), format="code")
        self.image_spec = get_spec(shape=IMAGE_SHAPE, format="image")
        self.optimizer, self.batch_size = OPTIMIZER, BATCH
        self.classifier_loss = CLASSIFIER_LOSS
        self.regresser_loss = REGRESSER_LOSS
        self.replay = []
        self.train()

    def train(self):
        for session_number in range(SESSIONS):
            self.step = 0
            self.task = choice(TASK_PREPPERS)(self)
            self.task.model = TaskModel(
                self, in_specs=self.task.in_specs, out_specs=self.task.out_specs)
            for episode_number in range(EPISODES):
                with tf.device('/gpu:0'):
                    self.run_episode()

    @tf.function
    def run_episode(self):
        observation = self.clean_observation(self.task.env.reset())
        done = False
        while not done:
            with tf.GradientTape() as tape:
                outputs = self.task.model(observation)
                if self.task.out_specs[0].format is "discrete":
                    outputs = int(outputs.numpy().item(0))
                observation, reward, done, _ = self.task.env.step(outputs)
                observation = self.clean_observation(observation)
                if not tf.is_tensor(reward):
                    reward = tf.convert_to_tensor(reward)
                losses = [-reward] + self.task.model.losses
            gradients = tape.gradient(
                losses, self.task.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.task.model.trainable_variables))
            sum_of_losses = tf.math.reduce_sum(losses)
            tf.print(self.step, sum_of_losses)
            tf.summary.scalar("sum of losses", sum_of_losses)
            self.step = self.step + 1
            # memory = [observation, outputs, losses, new_observation]
            # self.replay.append(memory)
            # observation = new_observation

            # log("", debug=True)
            # self.n_steps = self.n_steps + 1
            # log(f" step: {self.n_steps} --- task: {self.task.key}", debug=True)
            # n_gradients = sum([1 if v is not None else 0 for v in gradients])
            # n_variables = len(self.task.model.trainable_variables)
            # percent = get_percent(n_variables, n_gradients)
            # log(f" {percent} of variables have gradients", debug=True)
            # try:
            #     loss = add_up(loss)
            # except Exception as e:
            #     log('exception adding up loss', e)
            #     loss = loss.numpy().item(0)
            # free_energy = add_up(losses)
            # percent = get_percent(free_energy, loss)
            # log(f" {percent} task loss at {round(loss, 4)}", debug=True)
            # log(f" {round(free_energy, 4)} free energy", debug=True)

    def clean_observation(self, observation):
        if observation is None:
            return observation
        if not tf.is_tensor(observation):
            observation = tf.convert_to_tensor(observation, dtype=DTYPE)
        if observation.dtype is not DTYPE:
            observation = tf.cast(observation, DTYPE)
        if observation.shape[0] is not self.batch_size:
            observation = tf.expand_dims(observation, 0)
        return observation

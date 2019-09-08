# agent.py - handle multitask AI
# from keras_radam.training import RAdamOptimizer
from random import choice
import tensorflow as tf

from nature import TaskModel
from nurture import get_images
from tools import get_spec

CLASSIFIER_LOSS = tf.losses.categorical_crossentropy
OPTIMIZER = tf.keras.optimizers.SGD()
REGRESSER_LOSS = tf.losses.MSLE
TASK_PREPPERS = [get_images]
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
        self.hw = (IMAGE_SHAPE[0], IMAGE_SHAPE[1])
        self.practice()

    def practice(self):
        task = choice(TASK_PREPPERS)(self)
        task.graph, task.model = TaskModel(
            self, in_specs=task.in_specs, out_specs=task.out_specs)
        with tf.device("/gpu:0"):
            self.run_data_session(task.data, task.model, task.loss)

    @tf.function
    def run_data_session(self, data, model, loss_fn):
        for step, (image, label) in data.enumerate():
            with tf.GradientTape() as tape:
                prediction = model(image)
                loss = [*model.losses, loss_fn(label, prediction)]
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))
            tf.print(step, loss)

    # @tf.function
    # def run_env_episode(self):
    #     observation = self.task.env.reset()
    #     done = False
    #     while not done:
    #         with tf.GradientTape() as tape:
    #             outputs = self.task.model(observation)
    #             print("outputs", outputs)
    #             observation, reward, done, _ = self.task.env.step(outputs)
    #             losses = -reward
    #         gradients = tape.gradient(
    #             losses, self.task.model.trainable_variables)
    #         self.optimizer.apply_gradients(
    #             zip(gradients, self.task.model.trainable_variables))
    #         sum_of_losses = tf.math.reduce_sum(losses)
    #         tf.print(self.step, sum_of_losses)
    #         tf.summary.scalar("sum of losses", sum_of_losses)
    #         self.n_steps = self.n_steps + 1

    # def clean_observation(self, observation):
    #     if observation is None:
    #         return observation
    #     if not tf.is_tensor(observation):
    #         observation = tf.convert_to_tensor(observation, dtype=DTYPE)
    #     if observation.dtype is not DTYPE:
    #         observation = tf.cast(observation, DTYPE)
    #     if observation.shape[0] is not self.batch_size:
    #         observation = tf.expand_dims(observation, 0)
    #     return observation

    # def show_result(self, loss, losses, gradients):
    #     log("", debug=True)
    #     log(f" step: {self.n_steps} --- task: {self.task.key}", debug=True)
    #     n_gradients = sum([1 if v is not None else 0 for v in gradients])
    #     n_variables = len(self.task.model.trainable_variables)
    #     percent = get_percent(n_variables, n_gradients)
    #     log(f" {percent} of variables have gradients", debug=True)
    #     try:
    #         loss = add_up(loss)
    #     except Exception as e:
    #         log('exception adding up loss', e)
    #         loss = loss.numpy().item(0)
    #     free_energy = add_up(losses)
    #     percent = get_percent(free_energy, loss)
    #     log(f" {percent} task loss at {round(loss, 4)}", debug=True)
    #     log(f" {round(free_energy, 4)} free energy", debug=True)

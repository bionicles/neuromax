# agent.py - handle multitask AI
from random import choice
import tensorflow as tf

from tools import get_spec, get_uniform
from nature import TaskModel, Radam
from nurture import get_images
reduce_mean = tf.math.reduce_mean

REGRESSER_LOSS = tf.keras.losses.MeanSquaredLogarithmicError(
    reduction=tf.keras.losses.Reduction.NONE)
CLASSIFIER_LOSS = tf.keras.losses.CategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE)
MAE = tf.keras.losses.MeanAbsoluteError(
    reduction=tf.keras.losses.Reduction.NONE)
TASK_PREPPERS = [get_images]
OPTIMIZER = Radam()
DTYPE = tf.float32

IMAGE_SHAPE = (8, 8, 1)
CODE_SHAPE = (4, 8)
LOSS_SHAPE = (3,)
BATCH = 5


class Agent:
    """Entity which solves tasks using models made of bricks"""

    def __init__(self):
        self.image_spec = get_spec(shape=IMAGE_SHAPE, format="image")
        self.code_spec = get_spec(shape=CODE_SHAPE, format="code")
        self.loss_spec = get_spec(shape=LOSS_SHAPE, format="loss")
        self.classifier_loss = CLASSIFIER_LOSS
        self.regresser_loss = REGRESSER_LOSS
        self.optimizer = OPTIMIZER
        self.batch = BATCH
        self.practice()

    def practice(self):
        task = choice(TASK_PREPPERS)(self)
        task = self.add_specs(task)
        task.graph, task.model = TaskModel(
            self, in_specs=task.in_specs, out_specs=task.out_specs)
        with tf.device("/gpu:0"):
            self.run_data_session(task)

    def add_specs(self, task):
        in_specs, out_specs = list(task.in_specs), list(task.out_specs)
        in_specs += in_specs + out_specs + out_specs + [self.loss_spec] * 3
        out_specs += [self.loss_spec]
        task.in_specs, task.out_specs = in_specs, out_specs
        return task

    @tf.function
    def run_data_session(self, task):
        data, model, loss = task.data, task.model, task.loss
        last_image = tf.ones((BATCH, *IMAGE_SHAPE), tf.float32)
        last_pred = get_uniform(task.out_specs[0].shape, batch=self.batch)
        last_loss = tf.ones((BATCH, LOSS_SHAPE[0]), tf.float32)
        last_true = tf.ones_like(last_pred)
        value_pred = tf.ones_like(last_loss)
        criticism = tf.ones_like(last_loss)
        for step, (image, y_true) in data.enumerate():
            with tf.GradientTape() as tape:
                tape.watch([
                    model.trainable_variables,
                    image, last_image, last_true, last_pred,
                    value_pred, criticism, last_loss])
                outputs = model([
                    image, last_image, last_true, last_pred,
                    value_pred, criticism, last_loss])
                y_pred, value_pred, criticism = outputs
                class_loss = tf.expand_dims(loss(y_true, y_pred), -1)
                value_loss = MAE(class_loss, value_pred)
                criticism = criticism + value_pred
                critic_loss = MAE(class_loss, criticism)
                losses = [class_loss, value_loss, critic_loss, model.losses]
            grads = tape.gradient(losses, model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, model.trainable_variables))
            critic_loss = tf.expand_dims(critic_loss, -1)
            value_loss = tf.expand_dims(value_loss, -1)
            last_loss = tf.concat([class_loss, value_loss, critic_loss], -1)
            last_image = image
            last_true = y_true
            tf.print(
                step, reduce_mean(class_loss),
                "value:", reduce_mean(value_pred), reduce_mean(value_loss),
                "critic:", reduce_mean(criticism), reduce_mean(critic_loss))
        return "fuck"

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

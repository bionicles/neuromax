# agent.py - handle multitask AI
# from keras_radam import RAdam
from attrdict import AttrDict
import tensorflow as tf

from tools import get_percent, get_spec, add_up, log, show_model
import nature
import nurture

OPTIMIZER = tf.keras.optimizers.SGD(3e-4, clipvalue=0.04)
CLASSIFIER_LOSS = tf.losses.categorical_crossentropy
REGRESSER_LOSS = tf.losses.MSLE
EPISODES_PER_SESSION = 10000
USE_OTHER_LOSSES = True

IMAGE_SHAPE = (32, 32, 1)

BATCH = 1
ATOMS = 32
CHANNELS = 1

K = tf.keras
B, L = K.backend, K.layers


class Agent:
    """Entity which learns to solve tasks using bricks"""

    def __init__(self):
        self.code_spec = get_spec(shape=(ATOMS, CHANNELS), format="code")
        self.image_spec = get_spec(shape=IMAGE_SHAPE, format="image")
        self.classifier_loss = CLASSIFIER_LOSS
        self.regresser_loss = REGRESSER_LOSS
        self.batch_size = BATCH
        self.optimizer = OPTIMIZER
        self.n_gradient_steps = 0
        self.train()

    def train(self):
        self.pull_task()
        self.pull_model()
        for episode_number in range(EPISODES_PER_SESSION):
            self.run_episode()

    def pull_task(self):
        # prepper = nurture.prepare_env
        # if random.random() < 0.5:
        prepper = nurture.prepare_data
        self.task = prepper(self)

    def pull_model(self):
        self.task.model, self.task.roles = nature.use_graph_model(
            self, in_specs=self.task.in_specs, out_specs=self.task.out_specs)
        show_model(self.task.model, self.task.key)

    def clean_observation(self, observation):
        if not tf.is_tensor(observation):
            observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        if observation.dtype is not tf.float32:
            observation = tf.cast(observation, tf.float32)
        if observation.shape[0] is not self.batch_size:
            observation = tf.expand_dims(observation, 0)
        return observation

    def run_episode(self):
        observation = self.task.env.reset()
        observation = self.clean_observation(observation)
        done = False
        while not done:
            with tf.GradientTape() as tape:
                outputs = self.task.model(observation)
                normies, reconstructions, y_pred = self.unpack(
                    self.task.roles, outputs)
                if self.task.out_specs[0].format is "discrete":
                    y_pred = int(y_pred.numpy().item(0))
                observation, reward, done, _ = self.task.env.step(y_pred)
                if observation:
                    observation = self.clean_observation(observation)
                loss = -reward
                if not tf.is_tensor(loss):
                    loss = tf.convert_to_tensor(loss)
                losses = [loss]
                if USE_OTHER_LOSSES:
                    reconstruction_errors = self.compute_errors(
                        normies, reconstructions)
                    [losses.append(e) for e in reconstruction_errors]
            gradients = tape.gradient(
                losses, self.task.model.trainable_variables)
            gradients_and_variables = zip(
                gradients, self.task.model.trainable_variables)
            self.optimizer.apply_gradients(gradients_and_variables)

            log("", debug=True)
            self.n_gradient_steps += 1
            n_gradients = sum([1 if v is not None else 0 for v in gradients])
            n_variables = len(self.task.model.trainable_variables)
            free_energy = add_up(losses)
            log(f" step: {self.n_gradient_steps} --- task: {self.task.key}", debug=True)
            log(f" {get_percent(n_variables, n_gradients)} of variables have gradients", debug=True)
            if USE_OTHER_LOSSES:
                total_reconstruction_error = add_up(reconstruction_errors)
                log(f" {get_percent(free_energy, total_reconstruction_error)} reconstruction error at {round(total_reconstruction_error, 2)}", debug=True)
            try:
                loss = add_up(loss)
            except Exception as e:
                log('exception while adding up loss', e)
                loss = loss.numpy().item(0)
            log(f" {get_percent(free_energy, loss)} task loss at {round(loss, 2)}", debug=True)
            log(f" {round(free_energy, 1)} free energy", debug=True)

    @staticmethod
    def unpack(roles, outputs):
        unpacked = AttrDict()
        for n, (role, output) in enumerate(zip(roles, outputs)):
            if role not in unpacked.keys():
                unpacked[role] = [output]
            else:
                unpacked[role].append(output)
        return (unpacked.normie,
                unpacked.reconstruction,
                unpacked.action[0])

    def compute_errors(self, y_true_list, y_pred_list):
        errors = []
        for n, (y_true, y_pred) in enumerate(zip(y_true_list, y_pred_list)):
            errors.append(self.regresser_loss(y_true, y_pred))
        return errors

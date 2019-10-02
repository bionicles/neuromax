# agent.py - handle multitask AI
from random import choice
import tensorflow as tf
import optuna
from tools import get_spec, get_uniform, prettify, make_id, log
from nature import TaskModel, Radam
from nurture import get_images

DTYPE = tf.float32
B = tf.keras.backend
L = tf.keras.losses
REGRESSER_LOSS = L.MeanSquaredLogarithmicError(reduction=L.Reduction.NONE)
CLASSIFIER_LOSS = L.CategoricalCrossentropy(reduction=L.Reduction.NONE)
MAE = L.MeanAbsoluteError(reduction=L.Reduction.NONE)
TASK_PREPPERS = [get_images]
OPTIMIZER = Radam
BUFFER = 9001
STEPS = 420

# D_CODE_OPTIONS = [1, 2, 4, 8, 16, 32]
D_CODE_OPTIONS = [4]
IMAGE_SHAPE = (16, 16, 1)
CODE_SHAPE = [16, 4]
LOSS_SHAPE = (3,)
BATCH = 32


class Agent:
    """Entity which solves tasks using models made of bricks"""

    def __init__(self, trial):
        self.hp = trial
        CODE_SHAPE[-1] = self.pull("d_code", D_CODE_OPTIONS)
        self.image_spec = get_spec(shape=IMAGE_SHAPE, format="image")
        self.code_spec = get_spec(shape=CODE_SHAPE, format="code")
        self.loss_spec = get_spec(shape=LOSS_SHAPE, format="loss")
        self.classifier_loss = CLASSIFIER_LOSS
        self.regresser_loss = REGRESSER_LOSS
        self.optimizer = OPTIMIZER(self)
        self.objectives = []
        self.batch = BATCH
        self.practice()

    def practice(self):
        task = self.add_specs(choice(TASK_PREPPERS)(self))
        task.graph, task.model = TaskModel(
            self, in_specs=task.in_specs, out_specs=task.out_specs)
        params = tf.math.log(tf.cast(task.model.count_params(), tf.float32))
        data, model, loss = task.data, task.model, task.loss
        last_pred = get_uniform(task.out_specs[0].shape, batch=self.batch)
        with tf.device("/gpu:0"):
            objective = self.run_data_session(data, model, loss, last_pred)
        self.objective = objective * params
        log(f"OBJECTIVE {self.objective}", color="green", debug=True)

    def add_specs(self, task):
        in_specs, out_specs = list(task.in_specs), list(task.out_specs)
        in_specs += out_specs + out_specs + [self.loss_spec] * 3
        out_specs += [self.loss_spec]
        task.in_specs, task.out_specs = in_specs, out_specs
        return task

    @tf.function(experimental_relax_shapes=True)
    def run_data_session(self, data, model, loss, last_pred):
        last_loss = tf.ones((BATCH, LOSS_SHAPE[0]), tf.float32)
        v_pred, criticism = tf.ones_like(last_loss), tf.ones_like(last_loss)
        last_true = tf.ones_like(last_pred)
        objective = 0.
        for step, (image, y_true) in data.take(STEPS).enumerate():
            with tf.GradientTape() as tape:
                y_pred, v_pred, criticism = model([
                    image, last_true, last_pred, v_pred, criticism, last_loss])
                class_loss = tf.expand_dims(loss(y_true, y_pred), -1)
                v_loss = MAE(class_loss, v_pred)
                criticism = criticism + v_pred
                c_loss = MAE(class_loss, criticism)
                losses = [class_loss * 9001., v_loss, c_loss, model.losses]
            grads = tape.gradient(losses, model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, model.trainable_variables))
            c_loss = tf.expand_dims(c_loss, -1)
            v_loss = tf.expand_dims(v_loss, -1)
            last_loss = tf.concat([class_loss, v_loss, c_loss], -1)
            last_true, last_pred = tf.identity(y_true), tf.identity(y_pred)
            tf.print(
                step, "L:", prettify(class_loss),
                "P:", prettify(v_pred), prettify(criticism),
                "E:", prettify(v_loss), prettify(c_loss))
            objective = objective + tf.math.reduce_sum(last_loss)
        return objective

    def pull(self, *args, log_uniform=False, id=False):
        args = list(args)
        assert isinstance(args[0], str)
        if id:
            args[0] = make_id(args[0])
        if isinstance(args[1], list):
            # Categorical
            # opt = trial.suggest_categorical('optimizer', ['MomentumSGD', 'Adam'])
            return self.log_and_return(args, self.hp.suggest_categorical(*args))
        elif isinstance(args[1], int) and isinstance(args[2], int):
            # Int
            # num_layers = trial.suggest_int('num_layers', 1, 3)
            return self.log_and_return(args, self.hp.suggest_int(*args))
        elif isinstance(args[1], float) and isinstance(args[2], float):
            # Uniform
            # dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)
            # Loguniform
            # learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
            if log_uniform:
                return self.log_and_return(args, self.hp.suggest_loguniform(*args))
            return self.log_and_return(args, self.hp.suggest_uniform(*args))
        elif len(args) is 4:
            # Discrete-uniform
            # drop_path_rate = trial.suggest_discrete_uniform('drop_path_rate', 0.0, 1.0, 0.1)
            return self.log_and_return(args, self.hp.suggest_discrete_uniform(*args))
        else:
            log("FAILED TO PULL FOR ARGS", args, color="red")
            raise Exception("AI.Pull failed")

    @staticmethod
    def log_and_return(args, hp):
        log(f"HP {args[0]}", hp)
        return hp
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

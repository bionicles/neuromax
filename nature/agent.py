# agent.py - handle multitask AI
from attrdict import AttrDict
import tensorflow as tf
import numpy as np
import random

from .bricks.graph_model.graph_model import GraphModel
from .bricks.interface import Interface

from tools.map_attrdict import map_attrdict
from tools.get_percent import get_percent
from tools.show_model import show_model
from tools.get_spec import get_spec
from tools.add_up import add_up
from tools.log import log

SWITCH_TO_MAE_LOSS_WHEN_FREE_ENERGY_BELOW = 4.2
OPTIMIZER = tf.keras.optimizers.SGD(3e-4, clipvalue=0.04)
LOSS_FN = tf.keras.losses.MSLE

EPISODES_PER_PRACTICE_SESSION = 100

IMAGE_SHAPE = (32, 32, 4)
CODE_ATOMS = 32
BATCH_SIZE = 1

K = tf.keras
B = K.backend
L = K.layers


class Agent:
    """Entity which learns to solve tasks using bricks"""

    def __init__(self, tasks):
        self.n_gradient_steps = 0
        self.batch_size = BATCH_SIZE
        self.loss_fn = LOSS_FN
        self.tasks = tasks
        self.parameters = AttrDict({})
        self.code_spec = get_spec(shape=(CODE_ATOMS, 1), format="code")
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
        self.optimizer = OPTIMIZER

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
                prior = tf.random.normal(self.code_spec.shape)
                codes.append(prior)
        log("we pass codes to GraphModel:", color="green")
        graph_outputs = self.graph_model(codes)
        log("we make predictions and save them", color="green")
        predictions = []
        predictor = None
        for graph_output_number, graph_output in enumerate(graph_outputs):
            graph_out_with_code = L.Concatenate(1)([graph_output, task_code])
            log("graph_out_with_code shape", graph_out_with_code.shape,
                color="red", debug=1)
            if predictor is None:
                in_spec = get_spec(
                    format="code", shape=graph_out_with_code.shape[1:])
                predictor = Interface(
                    self, "predictor", in_spec, self.code_spec)
            prediction = predictor(graph_out_with_code)
            output_roles.append(f"prediction-{graph_output_number}")
            predictions.append(prediction)
            outputs.append(prediction)
        log("we assemble a world model for the actuators", color="green")
        predictions = [
            tf.expand_dims(p, 0) if len(p.shape) < 3 else p
            for p in predictions]
        world_model = tf.concat([*codes, *predictions], 1)
        world_model_spec = get_spec(format="code", shape=world_model.shape[1:])
        log("we pass the model to actuators to get actions", color="green")
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
        [log("input", n, list(i.shape), color="yellow")
         for n, i in enumerate(inputs)]
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
        if pkey in self.parameters.keys():
            return self.parameters[pkey]
        maybe_choice_or_choices = np.random.choice(
            options, size=n, p=p, replace=replace).tolist()
        self.parameters[pkey] = maybe_choice_or_choices
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
        if pkey in self.parameters.keys():
            return self.parameters[pkey]
        if n is None:
            parameter = method(shape, dtype=dtype)
        else:
            parameter = [method(shape, dtype=dtype) for _ in range(n)]
        self.parameters[pkey] = parameter
        return parameter

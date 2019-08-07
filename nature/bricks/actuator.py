# https://toolz.readthedocs.io/en/latest/api.html#dicttoolz might be fun

import tensorflow as tf

from .kernel_conv import KConvSet1D
from tools import concat_1D_coords

K = tf.keras
L = K.layers

# current outputs required:

# 2D -> onehot (clevr answer)
# 2D -> int (discrete control for MountainCar-v0)
# 2D -> 2D (ragged force field for protein dynamics)

ONEHOT_ACTIVATION_OPTIONS = ["sigmoid"]
RAGGED_ACTIVATION_OPTIONS = ["tanh"]
MIN_FILTERS, MAX_FILTERS = 32, 64
MIN_UNITS, MAX_UNITS = 32, 64


class Actuator:
    """
    Actuator resizes reshapes 1 input tensor into 1 output tensor
    because inputs have different sizes and shapes

    Args:
    agent: Agent which holds this brick and has pull_choices/pull_numbers
    task_key: string key to retrieve the task information
    in_spec: AttrDict description of input
    out_spec: AttrDict description of output
    """

    def __init__(self, agent, task_key, in_spec, out_spec, output_number):
        self.agent = agent
        self.pull_numbers = agent.pull_numbers
        self.pull_choices = agent.pull_choices
        print(f"Interface.__init__ {in_spec} ---> {out_spec}")
        self.output_number = output_number
        self.out_spec = out_spec
        self.task_key = task_key
        self.in_spec = in_spec
        self.name = f"{task_key}_{out_spec.format}_actuator_{output_number}"
        self.build(self.out_spec.format)

    def build(self, format):
        self.input = K.Input(self.in_spec.shape)
        if format == "onehot":
            self.d_out = self.out_spec.shape[-1]
            self.get_onehot_output()
        elif format == "discrete":
            self.d_out = self.out_spec.n
            self.get_discrete_output()
        elif format == "ragged":
            self.shape_variable_key = self.out_spec.shape[0]
            self.d_out = self.out_spec.shape[-1]
            self.get_ragged_output()
        self.model = K.Model(self.input, self.output)

    def get_onehot_output(self):
        """Return output to build a onehot actuator"""
        units = self.pull_numbers(f"{self.name}_units", MIN_UNITS, MAX_UNITS)
        activation = self.pull_choices(f"{self.name}_activation",
                                       ONEHOT_ACTIVATION_OPTIONS)
        output = L.Dense(units, activation)(self.input)
        output = L.Dense(self.d_out, activation)(output)
        self.output = K.activations.softmax(output)

    def get_discrete_output(self):
        """Return output to build a onehot actuator"""
        self.get_onehot_output()
        self.output = tf.argmax(self.output, axis=0)
        self.call = self.call_onehot_or_discrete

    def get_ragged_output(self):
        self.output = KConvSet1D(self.agent, self.in_spec, self.out_spec, 1)
        self.call = self.call_ragged

    def get_shape_var(self, key):
        return self.agent.current_task_dict.shape_variables[self.shape_variable_key].value

    def call_ragged(self, code):
        noise = tf.random.normal((self.get_shape_var(), self.d_out))
        noise = concat_1D_coords(noise, normalize=True)
        return self.model(code, noise)

    def call_onehot_or_discrete(self, code):
        return self.model(code)

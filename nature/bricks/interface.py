from tensorflow_addons import InstanceNormalization
import tensorflow_probability as tfp
import tensorflow as tf

from .kernel_conv import KConvSet1D
from tools import normalize

tfd = tfp.distributions
tfpl = tfp.layers
K = tf.keras
L = K.layers

# SENSORS:
#     image -> code (eyeball)
#     ragged -> code (NLP + atoms)

# ACTUATORS
#     code -> onehot (clevr answer)
#     code -> int (discrete control for MountainCar-v0)
#     code -> ragged (ragged force field for protein dynamics)
#     code -> image (reconstruct image)

ONEHOT_ACTIVATION_OPTIONS = ["sigmoid"]
RAGGED_ACTIVATION_OPTIONS = ["tanh"]
MIN_FILTERS, MAX_FILTERS = 32, 64
MIN_UNITS, MAX_UNITS = 32, 64


class Interface:
    """
    Interface resizes + reshapes + reformats in_spec into out_spec
    because inputs have different sizes and shapes

    Args:
        agent: Agent which holds this brick and has pull_choices/pull_numbers
        task_id: string key to retrieve task data
        in_spec: AttrDict description of input
        out_spec: AttrDict description of output

        specs hold info like (shape, n, format, high, low, dtype)
    """

    def __init__(self, agent, task_id, in_spec, out_spec):
        self.pull_numbers = agent.pull_numbers
        self.pull_choices = agent.pull_choices
        self.agent = agent
        print(f"Interface.__init__ {in_spec} ---> {out_spec}")
        self.task_id, self.out_spec, self.in_spec = task_id, in_spec, out_spec
        self.name = f"{task_id}_{in_spec}_{out_spec}_interface"
        self.input = K.Input(in_spec.shape)
        self.normie = InstanceNormalization()(self.input)
        format = out_spec.format
        if format == "onehot":
            self.get_onehot_output()
        elif format == "discrete":
            self.get_discrete_output()
        elif format == "ragged":
            self.get_ragged_output()
        self.model = K.Model(self.input, self.output)

    def get_onehot_actuator_output(self):
        units = self.pull_numbers(f"{self.name}_units", MIN_UNITS, MAX_UNITS)
        activation = self.pull_choices(f"{self.name}_activation",
                                       ONEHOT_ACTIVATION_OPTIONS)
        output = tfpl.DenseVariational(units, activation)(self.normie)
        output = tfpl.DenseVariational(units, activation)(output)
        d_out = self.out_spec.shape[-1] if self.d_out is None else self.d_out
        self.output = tfpl.OneHotCategorical(d_out)(output)

    def get_discrete_actuator_output(self):
        self.d_out = self.out_spec.n
        self.get_onehot_output()
        self.output = tf.argmax(self.output, axis=0)
        self.call = self.call_model

    def call_model(self, code):
        return self.model(code)

    def get_ragged_output(self):
        self.shape_variable_keys = [self.out_spec.shape[0]]
        self.d_out = self.out_spec.shape[-1]
        self.output = KConvSet1D(self.agent, self.in_spec, self.out_spec, 1)(self.input)
        self.call = self.call_ragged

    def get_shape_var(self, key):
        return self.agent.current_task_dict.shape_variables[key].value

    def call_ragged_actuator(self, code):
        shape_var_key = self.shape_variable_keys[0]
        ragged_dimension = self.get_shape_var(shape_var_key)
        coords = normalize(tf.range(ragged_dimension))  # need expand_dims?
        return self.model(code, coords)

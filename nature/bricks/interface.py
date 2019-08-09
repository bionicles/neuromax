import tensorflow_probability as tfp
import tensorflow_addons as tfa
import tensorflow as tf

from nature.bricks.multi_head_attention import MultiHeadAttention
from nature.bricks.vae import get_image_encoder_output, get_image_decoder_output
from nature.bricks.k_conv import KConvSet1D

from tools.concat_1D_coords import concat_1D_coords
from tools.concat_2D_coords import concat_2D_coords
from tools.normalize import normalize
from tools.get_size import get_size


InstanceNormalization = tfa.layers.InstanceNormalization

tfd = tfp.distributions
tfpl = tfp.layers
K = tf.keras
L = K.layers

# input2code
RAGGED_SENSOR_LAST_ACTIVATION_OPTIONS = ["tanh"]
ONEHOT_SENSOR_LAST_ACTIVATION_OPTIONS = ['sigmoid']
# code2code
CODE_INTERFACE_LAST_ACTIVATION_OPTIONS = ["tanh"]
CODE_INTERFACE_ACTIVATION_OPTIONS = ["tanh"]
# code2output
ONEHOT_ACTUATOR_ACTIVATION_OPTIONS = ["sigmoid"]
RAGGED_ACTUATOR_ACTIVATION_OPTIONS = ["tanh"]
# layers
MIN_FILTERS, MAX_FILTERS = 32, 64
MIN_UNITS, MAX_UNITS = 32, 64


class Interface:
    """
    Interface resizes + reshapes + reformats in_spec into out_spec
    because inputs have different sizes and shapes

    SENSORS:
        image -> code (eyeball)
        ragged -> code (NLP + atoms)
        onehot -> code (task number)

    MESSAGES:
        code -> code (internal message passing)

    ACTUATORS
        code -> onehot (clevr answer)
        code -> int (discrete control for MountainCar-v0)
        code -> ragged (ragged force field for protein dynamics)
        code -> image (reconstruct image)

    Args:
        agent: Agent which holds this brick and has pull_choices/pull_numbers
        task_id: string key to retrieve task data
        in_spec: AttrDict description of input
        out_spec: AttrDict description of output

        specs hold info like (shape, n, format, high, low, dtype)
    """

    def __init__(self, agent, task_id, in_spec, out_spec, input_number=None):
        self.pull_numbers = agent.pull_numbers
        self.pull_choices = agent.pull_choices
        self.agent = agent
        out_spec.size = get_size(out_spec.shape) if out_spec.format is "code" else None
        self.task_id, self.out_spec, self.in_spec = task_id, in_spec, out_spec
        self.brick_id = f"{task_id}_interface_{input_number}_{in_spec.format}_to_{out_spec.format}"
        print(f"Interface.__init__ -- {self.brick_id}")
        self.input_number = input_number
        self.shape_variable_key = None
        self.build_model()

    def add_coords_to_shape(shape):
        """Increase channels of a shape to account for coordinates"""
        new_shape = list(shape)
        new_shape[-1] += len(new_shape) - 1
        return tuple(new_shape)

    def build_model(self):
        if "add_coords" in self.in_spec.keys():
            self.in_spec.shape = self.add_coords_to_shape(self.in_spec.shape)
        self.input = K.Input(self.in_spec.shape)
        self.output = InstanceNormalization()(self.input)
        if self.in_spec.format is not "image":
            self.output = MultiHeadAttention(
                self.agent, self.brick_id)(self.output)
        in_spec, out_spec = self.in_spec, self.out_spec
        if in_spec.format is not "code" and out_spec.format is "code":
            model_type = f"{in_spec.format}_sensor"
        if in_spec.format is "code" and out_spec.format is "code":
            model_type = f"code_interface"
        if in_spec.format is "code" and out_spec.format is not "code":
            model_type = f"{out_spec.format}_actuator"
        builder_method = f"self.get_{model_type}_output()"
        try:
            eval(builder_method)
            if "ragged" in [in_spec.format, out_spec.format]:
                self.call = self.call_ragged
            else:
                self.call = self.call_model
        except Exception as e:
            print("INTERFACE FAILED TO BUILD!!!!!!!!!!!11111111one")
            print("builder_method", builder_method)
            print("in_spec", in_spec)
            print("out_spec", out_spec)
        self.model = K.Model(self.input, self.output)

    def get_image_sensor_output(self):
        self.output = get_image_encoder_output(
            self.agent, self.brick_id, self.output, self.out_spec.shape)

    def get_image_actuator_output(self):
        self.output = get_image_decoder_output(
            self.agent, self.brick_id, self.output, self.out_spec.shape)

    def get_ragged_sensor_output(self):
        output = L.Flatten()(self.output)
        activation = self.pull_choices(f"{self.brick_id}_last_activation",
                                       RAGGED_SENSOR_LAST_ACTIVATION_OPTIONS)
        output = tfpl.DenseVariational(self.out_size, activation)
        self.output = L.Reshape(self.out_spec.shape)(output)

    def get_ragged_actuator_output(self):
        if None in self.out_spec.shape:
            self.shape_variable_key = self.agent.get_shape_var_id(
                self.task_id, self.input_number, 0)
        self.d_out = self.out_spec.shape[-1]
        self.output = KConvSet1D(self.agent, self.in_spec, self.out_spec, 1)(self.output)

    def call_ragged(self, input):
        if self.shape_variable_key is None:
            input = concat_1D_coords(input, normalize=True)
        else:
            ragged_dimension = self.agent.parameters[self.shape_var_key]
            coords = normalize(tf.range(ragged_dimension))  # need expand_dims?
            input = tf.concat([input, coords], -1)
        return self.model(input)

    def get_code_interface_output(self):
        code_size = self.out_spec.size
        activation = self.pull_choices(f"{self.brick_id}_last_activation",
                                       CODE_INTERFACE_LAST_ACTIVATION_OPTIONS)
        self.output = tfpl.DenseVariational(code_size, activation)(self.output)

    def get_onehot_sensor_output(self):
        code_size = self.out_spec.size
        activation = self.pull_choices(f"{self.brick_id}_last_activation",
                                       ONEHOT_SENSOR_LAST_ACTIVATION_OPTIONS)
        self.output = tfpl.DenseVariational(code_size, activation)

    def get_onehot_actuator_output(self):
        units = self.pull_numbers(f"{self.brick_id}_units", MIN_UNITS, MAX_UNITS)
        activation = self.pull_choices(f"{self.name}_activation",
                                       ONEHOT_ACTUATOR_ACTIVATION_OPTIONS)
        output = tfpl.DenseVariational(units, activation)(self.output)
        d_out = self.out_spec.shape[-1] if self.d_out is None else self.d_out
        self.output = tfpl.OneHotCategorical(d_out)(output)

    def get_discrete_actuator_output(self):
        self.d_out = self.out_spec.n
        self.get_onehot_output()
        self.output = tf.argmax(self.output, axis=0)

    def call_model(self, input):
        if self.in_spec.format is "image":
            input = concat_2D_coords(input, normalize=True)
        else:
            input = concat_1D_coords(input, normalize=True)
        return self.model(input)

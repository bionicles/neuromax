import tensorflow_probability as tfp
import tensorflow_addons as tfa
import tensorflow as tf

from nature.bricks.vae import get_image_encoder_output, get_image_decoder_output
from nature.bricks.dense import get_dense_out
from nature.bricks.k_conv import KConvSet1D

from tools.concat_2D_coords import concat_2D_coords
from tools.normalize import normalize
from tools.get_spec import get_spec
from tools.get_size import get_size
from tools.log import log

InstanceNormalization = tfa.layers.InstanceNormalization

tfd = tfp.distributions
tfpl = tfp.layers
K = tf.keras
L = K.layers

# input2code
RAGGED_SENSOR_LAST_ACTIVATION_OPTIONS = ["tanh"]
ONEHOT_SENSOR_LAST_ACTIVATION_OPTIONS = ['sigmoid']
BOX_SENSOR_LAST_ACTIVATION_OPTIONS = ["sigmoid"]
# code2code
CODE_INTERFACE_LAST_ACTIVATION_OPTIONS = ["tanh"]
CODE_INTERFACE_ACTIVATION_OPTIONS = ["tanh"]
# code2output
ONEHOT_ACTUATOR_ACTIVATION_OPTIONS = ["sigmoid"]
RAGGED_ACTUATOR_ACTIVATION_OPTIONS = ["tanh"]
BOX_ACTUATOR_LAST_ACTIVATION_OPTIONS = ["linear"]
NORMAL_DISTRIBUTION_OPTIONS = [
    tfpl.MixtureNormal, tfpl.IndependentNormal, tfpl.MultivariateNormalTriL]


class Interface(L.Layer):
    """
    Interface resizes + reshapes + reformats in_spec into out_spec
    because inputs have different sizes and shapes

    Args:
        agent: Agent which holds this brick and has pull_choices/pull_numbers
        task_id: string key to retrieve task data
        in_spec: AttrDict description of input
        out_spec: AttrDict description of output

        specs hold info like (shape, n, format, high, low, dtype)

    SENSORS:
        image -> code (eyeball)
        ragged -> code (NLP + atoms)
        onehot -> code (task number)
        box -> code (bounded arrays)
        float -> code (loss values)

    MESSAGES:
        code -> code (internal message passing)

    ACTUATORS
        code -> onehot (clevr answer)
        code -> int (discrete control for MountainCar-v0)
        code -> ragged (ragged force field for protein dynamics)
        code -> image (reconstruct image)
        code -> box (bounded arrays)
    """

    def __init__(self, agent, task_id, in_spec, out_spec, input_number=None):
        self.pull_numbers = agent.pull_numbers
        self.pull_choices = agent.pull_choices
        self.agent = agent
        try:
            out_spec.size = get_size(out_spec.shape)
        except Exception as e:
            print("couldn't get size for out_spec", e, "\n")
        self.task_id, self.in_spec, self.out_spec = task_id, in_spec, out_spec
        self.brick_id = f"{task_id}_interface_{input_number}_{in_spec.format}_to_{out_spec.format}"
        print("")
        print(f"Interface.__init__ -- {self.brick_id}")
        print("in_spec", in_spec)
        print("out_spec", out_spec)
        self.input_layer_number = input_number
        self.shape_variable_key = None
        self.reshape = L.Reshape(self.out_spec.shape)
        if self.in_spec.format == "image":
            self.channels_before_concat_coords = self.in_spec.shape[-1]
            self.size_to_resize_to = self.get_hw(self.in_spec.shape)
            self.in_spec.shape = self.add_coords_to_shape(self.in_spec.shape)
            self.channels_after_concat_coords = self.in_spec.shape[-1]
            self.channel_changers = {}
        super(Interface, self).__init__()


    def build(self):
        self.input_layer = K.Input(self.in_spec.shape)
        self.normie = InstanceNormalization()(self.input_layer)
        self.out = self.normie
        self.call = self.call_model  # might be overridden in builder fn
        in_spec, out_spec = self.in_spec, self.out_spec
        if in_spec.format is not "code" and out_spec.format is "code":
            model_type = f"{in_spec.format}_sensor"
        if in_spec.format is "code" and out_spec.format is "code":
            model_type = f"code_interface"
        if in_spec.format is "code" and out_spec.format is not "code":
            model_type = f"{out_spec.format}_actuator"
        print(f"{model_type} interface input layer", self.input_layer)
        builder_fn = f"self.get_{model_type}_output()"
        eval(builder_fn)
        if "sensor" in model_type:
            self.out = [self.normie, self.out]
        self.model = K.Model(self.input_layer, self.out)

    @staticmethod
    def get_hw(shape):
        """Returns height and width of image shape (no batch dim)"""
        return (shape[0], shape[1])

    @staticmethod
    def add_coords_to_shape(shape):
        """Increase channels of a shape to account for coordinates"""
        new_shape = list(shape)
        new_shape[-1] += len(new_shape) - 1
        return tuple(new_shape)

    def get_image_sensor_output(self):
        self.call = self.call_image_sensor
        self.out = get_image_encoder_output(
            self.agent, self.brick_id, self.out, self.out_spec)
        self.make_normal()

    def get_image_actuator_output(self):
        self.out = get_image_decoder_output(
            self.agent, self.brick_id, self.out, self.out_spec)
        self.make_normal()

    def add_channel_changer(self, channels_in):
        """add a model to change the last dimension of the image"""
        if channels_in != self.channels_before_concat_coords:
            if channels_in not in self.channel_changers.keys():
                input_shape = tuple(*self.size_to_resize_to,  channels_in)
                channel_changer = K.Sequential([
                    L.Conv2D(self.channels_before_concat_coords, 1,
                             input_shape=input_shape)])
                self.channel_changers[channels_in] = channel_changer

    def call_image_sensor(self, input):
        image = tf.image.resize(input, self.size_to_resize_to)
        channels_in = tf.shape(image)[-1]
        if channels_in is not self.channels_before_concat_coords:
            image = self.channel_changers[channels_in](image)
        image = concat_2D_coords(image)
        return self.model(image)

    def get_ragged_sensor_output(self):
        """each input element innervates all output elements (one for all)"""
        self.out_placeholder_input = K.Input(self.out_spec.shape)
        self.input_layer = [self.input_layer, self.out_placeholder_input]
        self.call = self.call_ragged_sensor
        self.out = KConvSet1D(
            self.agent, self.brick_id, self.in_spec, self.out_spec,
            "one_for_all")([self.out, self.out_placeholder_input])
        self.make_normal()

    def call_ragged_sensor(self, input):
        code_placeholder = tf.random.normal(self.out_spec.shape)
        return self.model([input, code_placeholder])

    def get_ragged_actuator_output(self):
        """all input elements innervate each output element (all for one)"""
        self.shape_var_key = self.out_spec.variables[0][0]
        self.normalized_output_coords = K.Input((None, 1))
        self.input_layer = [self.normalized_output_coords, self.input_layer]
        self.call = self.call_ragged_actuator
        doubled_out_shape = self.out_spec.shape
        doubled_out_shape *= 2
        doubled_out_spec = get_spec(shape=doubled_out_shape,
                                    format="ragged")
        self.out = KConvSet1D(
            self.agent, self.brick_id, self.in_spec, doubled_out_spec,
            "all_for_one")([self.normalized_output_coords, self.out])

    def call_ragged_actuator(self, input):
        ragged_dimension = self.agent.parameters[self.shape_var_key]
        normalized_output_coords = normalize(tf.range(ragged_dimension))
        output = self.model([normalized_output_coords, input])
        loc, scale = tf.split(output, 2, axis=-1)
        return tfp.distributions.Normal(loc, tf.math.abs(scale))

    def get_box_sensor_output(self):
        self.flatten_resize_reshape()
        self.make_normal()

    def get_box_actuator_output(self):
        self.flatten_resize_reshape()
        self.make_normal()

    def get_float_sensor_output(self):
        self.flatten_resize_reshape()
        self.make_normal()

    def get_float_actuator_output(self):
        self.flatten_resize_reshape()
        self.make_normal()

    def get_code_interface_output(self):
        self.flatten_resize_reshape()
        self.make_normal()

    def get_onehot_sensor_output(self):
        self.flatten_resize_reshape()
        self.make_normal()

    def get_onehot_actuator_output(self):
        self.flatten_resize_reshape()
        self.make_categorical()

    def get_discrete_actuator_output(self):
        self.flatten_resize_reshape()
        self.make_categorical()

    def flatten_resize_reshape(self):
        if len(self.out.shape) > 2:
            self.out = L.Flatten()(self.out)
        out = get_dense_out(self.agent, self.brick_id, self.out,
                            units=self.out_spec.size)
        self.out = self.reshape(out)

    def make_categorical(self):
        self.out = tfpl.DenseReparameterization(
            tfpl.OneHotCategorical.params_size(self.out_spec.size)
            )(self.out)
        self.out = tfpl.OneHotCategorical(self.out_spec.size)(self.out)

    def make_normal(self):
        self.out = tfpl.DenseReparameterization(
            tfpl.IndependentNormal.params_size(self.out_spec.shape)
            )(self.out)
        self.out = tfpl.IndependentNormal(self.out_spec.shape)(self.out)

    def call_model(self, input):
        return self.model(input)

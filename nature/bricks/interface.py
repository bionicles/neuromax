import tensorflow_probability as tfp
import tensorflow_addons as tfa
import tensorflow as tf

from nature.bricks.vae import get_image_encoder_output, get_image_decoder_output
from nature.bricks.dense import get_dense_out
from nature.bricks.k_conv import KConvSet1D

from tools.concat_2D_coords import concat_2D_coords
from tools.get_spec import get_spec
from tools.get_size import get_size
from tools.log import log

InstanceNormalization = tfa.layers.InstanceNormalization

tfd = tfp.distributions
tfpl = tfp.layers
K = tf.keras
L = K.layers


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
        image -> code (eyeball / encoder)
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
        super(Interface, self).__init__()
        self.pull_numbers = agent.pull_numbers
        self.pull_choices = agent.pull_choices
        self.agent = agent
        self.task_id, self.in_spec, self.out_spec = task_id, in_spec, out_spec
        self.brick_id = f"{task_id}_interface_{input_number}_{in_spec.format}_to_{out_spec.format}"
        print("")
        print(f"Interface.__init__ -- {self.brick_id}")
        print("in_spec", in_spec)
        print("out_spec", out_spec)
        self.input_layer_number = input_number
        self.shape_variable_key = None
        if self.out_spec.format is not "code":
            self.reshape = L.Reshape(self.out_spec.shape)
        if self.in_spec.format == "image":
            self.channels_before_concat_coords = self.in_spec.shape[-1]
            self.size_to_resize_to = self.get_hw(self.in_spec.shape)
            encoder_shape = self.add_coords_to_shape(self.in_spec.shape)
            self.in_spec = get_spec(format="image", shape=encoder_shape)
            self.in_spec.size = get_size(encoder_shape)
        super(Interface, self).__init__()
        self.build()

    def build(self):
        if self.built:
            return
        self.input_layer = K.Input(self.in_spec.shape,
                                   batch_size=self.agent.batch_size)
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
        if ("sensor" in model_type and "loss" not in self.brick_id and "task" not in self.brick_id):
            if "image" in model_type:
                self.normie = tf.slice(self.normie,
                                       [0, 0, 0, 0], [-1, -1, -1, 4])
            self.out = [self.normie, self.out]
        self.model = K.Model(self.input_layer, self.out)
        self.built = True

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
        self.flatten_resize_reshape()
        self.make_normal()

    def call_image_sensor(self, input):
        image = tf.image.resize(input, self.size_to_resize_to)
        channels_in = int(image.shape[-1])
        if channels_in is not self.channels_before_concat_coords:
            channels_needed = self.channels_before_concat_coords - channels_in
            padding_shape = (self.agent.batch_size, self.in_spec.shape[0],
                             self.in_spec.shape[1], channels_needed)
            zeros = tf.zeros(padding_shape)
            image = tf.concat([image, zeros], -1)
        image = concat_2D_coords(image)
        return self.model(image)

    def get_image_actuator_output(self):
        self.out = get_image_decoder_output(
            self.agent, self.brick_id, self.out, self.out_spec)
        self.make_normal()

    def get_ragged_sensor_output(self):
        """each input element innervates all output elements (one for all)"""
        self.call = self.call_ragged_sensor
        self.out = KConvSet1D(
            self.agent, self.brick_id, self.in_spec, self.out_spec,
            "one_for_all")(self.out)
        self.flatten_resize_reshape()
        self.make_normal()

    def call_ragged_sensor(self, inputs):
        print("call_ragged_sensor")
        if "variables" in self.in_spec.keys():
            for id, n, index in self.in_spec.variables:
                self.agent.parameters[id] = inputs[n].shape[index]
        return self.call_model(inputs)

    def get_ragged_actuator_output(self):
        """all input elements innervate each output element (all for one)"""
        self.shape_var_id = self.out_spec.variables[0][0]
        self.placeholder = K.Input(
            (None, 1), batch_size=self.agent.batch_size)
        self.input_layer = [self.placeholder, self.input_layer]
        self.call = self.call_ragged_actuator
        doubled_out_shape = list(self.out_spec.shape)
        doubled_out_shape[-1] *= 2
        doubled_out_spec = get_spec(shape=doubled_out_shape, format="ragged")
        log("get_ragged_actuator_output", self.in_spec, doubled_out_spec,
            color="red")
        self.out = KConvSet1D(
            self.agent, self.brick_id, self.in_spec, doubled_out_spec,
            "all_for_one")([self.placeholder, self.out])

    def call_ragged_actuator(self, inputs):
        log("call_ragged_actuator", color="yellow")
        return self.model(inputs)
        # loc, scale = tf.split(output, 2, axis=-1)
        # return tfp.distributions.Normal(loc, tf.math.abs(scale))

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
        if self.out_spec.format is not "code":
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
        out_shape = self.out_spec.shape
        self.out = tfpl.IndependentNormal(out_shape)(self.out)

    def call_model(self, input):
        return self.model(input)

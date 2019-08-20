import tensorflow as tf

from nanoid import generate

from . import get_image_encoder_out, get_image_decoder_out
from . import get_norm_preact_out
from . import KConvSet1D
from . import get_multiply_out
from layers import get_dense_out
from helpers import swish

from tools import concat_coords
from tools import get_spec
from tools import get_size
from tools import get_hw
from tools import log

K = tf.keras
L = K.layers

UNITS, FN = 32, swish


# DEPRECATED -- switching to simpler "use_interface" function
class Interface(L.Layer):
    """
    Interface resizes + reshapes + reformats in_spec into out_spec
    because inputs have different sizes and shapes

    Args:
        agent: Agent which holds this brick and has pull_choices/pull_numbers
        task_id: string key to retrieve task data
        in_spec: AttrDict description of input
        out_spec: AttrDict description of out

        specs hold info like (shape, n, format, high, low, dtype)

    SENSORS:
        image -> code (eyeball / encoder)
        ragged -> code (NLP + atoms)
        onehot -> code (task number)
        box -> code (bounded arrays)
        float -> code (loss values)

    MESSAGES:
        code -> code (pass messages + predict)

    ACTUATORS
        code -> onehot (clevr answer)
        code -> int (discrete control for MountainCar-v0)
        code -> ragged (ragged force field for protein dynamics)
        code -> image (reconstruct image)
        code -> box (bounded arrays)
    """

    def __init__(self, agent, task_id, in_spec, out_spec, in_number=None):
        super(Interface, self).__init__()
        self.pull_numbers = agent.pull_numbers
        self.pull_choices = agent.pull_choices
        self.agent = agent
        self.probabilistic = agent.probabilistic
        self.task_id, self.in_spec, self.out_spec = task_id, in_spec, out_spec
        self.brick_id = f"{task_id}{'_'+str(in_number) if in_number else ''}_{in_spec.format}_to_{out_spec.format}_{generate()}"
        self.debug = 1 if "predictor" in self.brick_id else 0
        log("", debug=self.debug)
        log(f"Interface.__init__ -- {self.brick_id}", debug=self.debug)
        log("in_spec", in_spec, debug=self.debug)
        log("out_spec", out_spec, debug=self.debug)
        self.input_layer_number = in_number
        self.shape_variable_key = None
        if self.in_spec.format == "image":
            self.channels_before_concat_coords = self.in_spec.shape[-1]
            self.size_to_resize_to = get_hw(self.in_spec.shape)
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
        log(f"{self.brick_id} input layer", self.input_layer, debug=self.debug)
        try:
            self.normie = get_norm_preact_out(self.agent, self.id, self.input_layer)
        except Exception as e:
            log("instance norm failed", e, color="red")
            self.normie = None
        self.out = self.normie if self.normie else self.input_layer
        self.call = self.call_model  # might be overridden in builder fn
        in_spec, out_spec = self.in_spec, self.out_spec
        if in_spec.format is not "code" and out_spec.format is "code":
            model_type = f"{in_spec.format}_sensor"
        if in_spec.format is "code" and out_spec.format is "code":
            model_type = f"code_interface"
        if in_spec.format is "code" and out_spec.format is not "code":
            model_type = f"{out_spec.format}_actuator"
        builder_fn = f"self.get_{model_type}_out()"
        eval(builder_fn)
        if ("sensor" in model_type and "loss" not in self.brick_id and "task" not in self.brick_id):
            if "image" in model_type:
                self.normie = tf.slice(self.normie,
                                       [0, 0, 0, 0], [-1, -1, -1, 4])
            outs = [self.normie, self.out]
        else:
            outs = self.out
        log(f"{self.brick_id} outs", outs, color="yellow")
        self.model = K.Model(self.input_layer, outs)
        self.built = True

    def make_categorical(self):
        self.flatten_resize_reshape()
        self.out = L.Activation("softmax")(self.out)

    def flatten_resize_reshape(self):
        log(f"flatten_resize_reshape {self.out.shape}-->{self.out_spec.shape}")
        out_size = self.out_spec.size
        out = L.Lambda(lambda x: concat_coords(x))(self.out)
        out = L.Flatten()(out)
        log(0, out.shape, color="yellow")
        if len(out.shape) is 3:
            out = get_multiply_out(self.agent, self.brick_id, out_size, out)
        else:
            out = get_dense_out(
                self.agent, self.brick_id, units=out_size, fn=FN)(out)
        log(5, out.shape, color="yellow")
        self.out = L.Reshape(self.out_spec.shape)(out)

    @staticmethod
    def add_coords_to_shape(shape):
        """Increase channels of a shape to account for coordinates"""
        new_shape = list(shape)
        new_shape[-1] += len(new_shape) - 1
        return tuple(new_shape)

    def get_image_sensor_out(self):
        self.call = self.call_image_sensor
        self.out = get_image_encoder_out(
            self.agent, self.brick_id, self.out, self.out_spec)
        self.flatten_resize_reshape()

    def call_image_sensor(self, input):
        image = tf.image.resize(input, self.size_to_resize_to)
        channels_in = int(image.shape[-1])
        if channels_in is not self.channels_before_concat_coords:
            channels_needed = self.channels_before_concat_coords - channels_in
            padding_shape = (self.agent.batch_size, self.in_spec.shape[0],
                             self.in_spec.shape[1], channels_needed)
            zeros = tf.zeros(padding_shape)
            image = tf.concat([image, zeros], -1)
        image = concat_coords(image)
        return self.model(image)

    def get_image_actuator_out(self):
        self.out = get_image_decoder_out(
            self.agent, self.brick_id, self.out, self.out_spec)
        self.flatten_resize_reshape()

    def get_ragged_sensor_out(self):
        """each input element innervates all out elements (one for all)"""
        self.out = KConvSet1D(
            self.agent, self.brick_id, self.in_spec, self.out_spec,
            "one_for_all")(self.out)
        self.flatten_resize_reshape()

    def get_ragged_actuator_out(self):
        """all input elements innervate each out element (all for one)"""
        self.placeholder = K.Input(
            (None, 1), batch_size=self.agent.batch_size)
        self.input_layer = [self.input_layer, self.placeholder]
        self.out = KConvSet1D(
            self.agent, self.brick_id, self.in_spec, self.out_spec,
            "all_for_one")([self.out, self.placeholder])
        log("get_ragged_actuator_out", self.out, color="red")

    def get_box_sensor_out(self):
        self.flatten_resize_reshape()

    def get_box_actuator_out(self):
        self.flatten_resize_reshape()

    def get_float_sensor_out(self):
        self.flatten_resize_reshape()

    def get_float_actuator_out(self):
        self.flatten_resize_reshape()

    def get_code_interface_out(self):
        self.flatten_resize_reshape()

    def get_onehot_sensor_out(self):
        self.flatten_resize_reshape()

    def get_onehot_actuator_out(self):
        self.make_categorical(one_hot=True)

    def get_discrete_actuator_out(self):
        self.make_categorical()

    def call_model(self, input):
        # log("")
        # log("call interface", self.brick_id, color="yellow")
        # log("in_spec", list(self.in_spec.shape) if not isinstance(self.in_spec.shape, int) else self.in_spec.shape, self.in_spec.format, color="blue")
        # if isinstance(input, list):
            # [log("input.shape", list(i.shape), color="yellow") for i in input]
        # else:
            # log("input.shape", list(input.shape), color="yellow")
        out = self.model(input)
        # log("out_spec", list(self.out_spec.shape), self.out_spec.format, color="blue")
        # if isinstance(out, list):
        #     [log("out", list(o.shape), color="yellow") for o in out]
        # else:
        #     log("out", list(out.shape), color="yellow")
        return out

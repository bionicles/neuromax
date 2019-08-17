import tensorflow_probability as tfp
import tensorflow_addons as tfa
import tensorflow as tf

from nature.bricks.vae import get_image_encoder_output, get_image_decoder_output
from nature.bricks.k_conv import KConvSet1D

from tools.concat_2D_coords import concat_2D_coords
from tools.get_prior import get_prior
from tools.get_spec import get_spec
from tools.get_size import get_size
from tools.log import log

InstanceNormalization = tfa.layers.InstanceNormalization

tfd = tfp.distributions
tfpl = tfp.layers
K = tf.keras
L = K.layers

UNITS, FN = 32, "tanh"


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
        self.brick_id = f"{task_id}{'_'+str(input_number) if input_number else ''}_{in_spec.format}_to_{out_spec.format}"
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
        log(f"input layer", self.input_layer)
        if self.in_spec.format is not "code":
            try:
                self.normie = InstanceNormalization()(self.input_layer)
                self.out = self.normie
            except Exception as e:
                log("instance norm failed", e, color="red")
        else:
            self.out = self.input_layer
        self.call = self.call_model  # might be overridden in builder fn
        in_spec, out_spec = self.in_spec, self.out_spec
        if in_spec.format is not "code" and out_spec.format is "code":
            model_type = f"{in_spec.format}_sensor"
        if in_spec.format is "code" and out_spec.format is "code":
            model_type = f"code_interface"
        if in_spec.format is "code" and out_spec.format is not "code":
            model_type = f"{out_spec.format}_actuator"
        builder_fn = f"self.get_{model_type}_output()"
        eval(builder_fn)
        if ("sensor" in model_type and "loss" not in self.brick_id and "task" not in self.brick_id):
            if "image" in model_type:
                self.normie = tf.slice(self.normie,
                                       [0, 0, 0, 0], [-1, -1, -1, 4])
            outputs = [self.normie, self.out]
        else:
            outputs = self.out
        log("outputs", outputs, color="yellow")
        self.model = K.Model(self.input_layer, outputs)
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
        self.make_normal()

    def call_ragged_sensor(self, inputs):
        if "variables" in self.in_spec.keys():
            for id, n, index in self.in_spec.variables:
                self.agent.parameters[id] = inputs[n].shape[index]
        return self.call_model(inputs)

    def get_ragged_actuator_output(self):
        """all input elements innervate each output element (all for one)"""
        self.shape_var_id = self.out_spec.variables[0][0]
        self.placeholder = K.Input(
            (None, 1), batch_size=self.agent.batch_size)
        self.input_layer = [self.input_layer, self.placeholder]
        self.out = KConvSet1D(
            self.agent, self.brick_id, self.in_spec, self.out_spec,
            "all_for_one")([self.out, self.placeholder])

    def get_box_sensor_output(self):
        self.make_normal()

    def get_box_actuator_output(self):
        self.make_normal()

    def get_float_sensor_output(self):
        self.make_normal()

    def get_float_actuator_output(self):
        self.make_normal()

    def get_code_interface_output(self):
        self.make_normal()

    def get_onehot_sensor_output(self):
        self.make_normal()

    def get_onehot_actuator_output(self):
        self.make_categorical(one_hot=True)

    def get_discrete_actuator_output(self):
        self.make_categorical()

    def make_categorical(self, units=UNITS, fn=FN, one_hot=False):
        prior_type = "one_hot_categorical" if one_hot else "categorical"
        prior, n = get_prior(self.out_spec.shape, prior_type=prior_type)
        out = L.Flatten()(self.out)
        out = tfpl.DenseReparameterization(units, fn)(out)
        out = tfpl.DenseReparameterization(units, fn)(out)
        p = tfpl.DenseReparameterization(n)(out)
        distribution = tfd.OneHotCategorical if one_hot else tfd.Categorical
        self.out = tfpl.DistributionLambda(
            make_distribution_fn=lambda p: distribution(
                probs=p, name=f"{prior_type}_{self.brick_id}"),
            # convert_to_tensor_fn=lambda d: d.sample(),
            # activity_regularizer=tfpl.KLDivergenceRegularizer(prior)
        )(p)

    def make_normal(self, units=UNITS, fn=FN):
        prior, shapes = get_prior(self.out_spec.shape, "normal")
        last_layer_units = int(get_size(shapes["loc"]))
        out = L.Flatten()(self.out)
        if out.shape[-1] is None:
            out = tf.einsum('ij->ji', out)
        out = tfpl.DenseReparameterization(units, fn)(out)
        out = tfpl.DenseReparameterization(units, fn)(out)
        loc_layer = tfpl.DenseReparameterization(last_layer_units)
        scale_layer = tfpl.DenseReparameterization(last_layer_units)
        loc, scale = loc_layer(out), scale_layer(out)
        if self.out_spec.shape[-1] is 1:
            expand = L.Lambda(lambda x: tf.expand_dims(x, -1))
            loc, scale = expand(loc), expand(scale)
        else:
            reshape = L.Reshape(shapes["loc"])
            loc, scale = reshape(loc), reshape(scale)
        self.out = tfpl.DistributionLambda(
                make_distribution_fn=lambda x: tfd.Normal(
                    loc=x[0], scale=tf.math.abs(x[1]),
                    name=f"N_{self.brick_id}"
                ),
                # convert_to_tensor_fn=lambda d: d.sample(),
                # activity_regularizer=tfpl.KLDivergenceRegularizer(prior)
            )([loc, scale])

    def call_model(self, input):
        print("")
        log("call interface", self.brick_id, color="blue")
        log("in_spec", self.in_spec, color="blue")
        log("out_spec", self.out_spec, color="blue")
        log("input", input, color="yellow")
        return self.model(input)

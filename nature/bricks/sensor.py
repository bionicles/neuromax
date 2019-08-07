import tensorflow_probability as tfp
import tensorflow as tf

from tools import add_distribution
from . import get_instance_normalizer
from . import KConvSet1D

tfd = tfp.distributions
tfpl = tfp.layers
K = tf.keras
L = K.layers

UNITS, ACTIVATION = 128, "tanh"


class Sensor:
    """
    (input) ---> (normie, code, reconstruction)

    Args:
    agent: Agent which holds this brick and has pull_choices/pull_numbers
    in_spec: an AttrDict with shape
    code_spec: an AttrDict with shape

    Returns:
    normie: normalized input
    code: latent code
    reconstruction: of the INSTANCE NORMALIZED input
    """

    def __init__(self, agent, in_spec, code_spec):
        self.agent = agent
        self.pull_numbers = agent.pull_numbers
        self.pull_choices = agent.pull_choices
        if "size" not in code_spec.keys() or code_spec.size is None:
            code_spec.size = tf.reduce_prod(code_spec.shape)
        self.code_spec = code_spec
        self.in_spec = in_spec
        self.normalizer = get_instance_normalizer(self.in_spec.shape)
        # we build the encoder
        self.encoder = K.Sequential([K.Input(self.in_spec.shape)])
        if in_spec.sensor_type is "dense":
            self.encoder.add(L.Dense(UNITS, ACTIVATION))
        elif in_spec.sensor_type is "lstm":
            self.encoder.add(L.LSTM(UNITS, ACTIVATION, return_sequences=True))
        self.encoder.add(L.Flatten())
        self.encoder.add(L.Dense(self.code_spec.size))
        self.encoder = add_distribution(self.encoder, self.code_spec.size)
        # we build the decoder
        decoder_code_input = K.Input(self.code_spec.shape)
        reconstruction = L.Dense(UNITS, ACTIVATION)(decoder_code_input)
        # to know size of variable shape inputs, we sample noise at runtime
        # and use autoregressive KConvSet1D
        if None in self.in_spec.shape:
            decoder_noise_input = K.Input(self.in_shape)
            reconstruction = KConvSet1D(agent, )(reconstruction, decoder_noise_input)
            inputs = [decoder_code_input, decoder_noise_input]
        else:  # we know the shape and use a dense layer, YOLO
            units = tf.math.reduce_prod(self.in_shape)
            reconstruction = L.Dense(units, ACTIVATION)
            inputs = [decoder_code_input]
        reconstruction = L.Reshape(self.in_spec.shape)(reconstruction)
        self.decoder = K.Model(inputs, reconstruction)

    def call(self, inputs):
        normie = self.normalizer(inputs)
        code = self.encoder(normie)
        if None in self.in_spec.shape:
            noise = tf.random.normal(tf.shape(self.inputs))
            reconstruction = self.decoder(code, noise)
        else:
            reconstruction = self.decoder(code)
        return normie, code, reconstruction

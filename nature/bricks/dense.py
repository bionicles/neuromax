import tensorflow_probability as tfp
import tensorflow as tf

from .activations import clean_activation, swish
from .noisedrop import NoiseDrop
from .chaos import EdgeOfChaos
from tools.log import log

K = tf.keras
B, L = K.backend, K.layers
tfpl = tfp.layers

UNITS = 256
TFP_LAYER = tfpl.DenseReparameterization
LAYER = NoiseDrop
FN = swish
L1, L2 = 0.001, 0.001
KERNEL_STDDEV = 2.952
BIAS_STDDEV = 0.04


def get_dense(agent, brick_id, layer=None, units=UNITS, fn=FN):
    if layer is None:
        layer = TFP_LAYER if agent.probabilistic else LAYER
    log("getting a dense layer for", brick_id, "units:", units, "fn:", fn)
    if fn is not None:
        fn = clean_activation(fn)
    return layer(
        units, activation=fn,
        kernel_regularizer=K.regularizers.L1L2(l1=L1, l2=L2),
        kernel_initializer=EdgeOfChaos(True, "swish"),
        bias_initializer=EdgeOfChaos(False, "swish")
        )

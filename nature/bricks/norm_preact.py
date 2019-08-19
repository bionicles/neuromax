from tensorflow_addons.layers import InstanceNormalization
import tensorflow as tf

from tools.get_brick import get_brick

from .helpers.activations import clean_activation, swish

K = tf.keras
L = K.layers

NORM = "instance"
FN = swish


def get_norm_preact_out(
        agent, id, out, norm=NORM, fn=FN, layer_fn=None, return_brick=False):
    if not norm and not fn:
        return out
    if norm is "instance":
        normalizer = InstanceNormalization()
    elif norm is "batch":
        normalizer = L.BatchNormalization()

    if fn is None:
        def op(out):
            return normalizer(out)
    else:
        activation = L.Activation(clean_activation(fn))

        def op(out):
            out = normalizer()(out)
            return activation(out)
    return get_brick(op, out, return_brick)

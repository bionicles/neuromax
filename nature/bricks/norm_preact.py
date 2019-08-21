from tensorflow_addons.layers import InstanceNormalization
import tensorflow as tf

from nature import clean_activation, swish
from tools import make_uuid

K = tf.keras
L = K.layers

NORM = "instance"
FN = swish


def use_norm_preact(
        agent, id, out, norm=NORM, fn=FN, layer_fn=None, return_brick=False,
        return_normie=False):
    id = make_uuid([id, "norm_preact"])

    if not norm and not fn:
        return out
    if norm is "instance":
        normalizer = InstanceNormalization()
    elif norm is "batch":
        normalizer = L.BatchNormalization()
    parts = dict(normalizer=normalizer)
    if fn:
        activation = L.Activation(clean_activation(fn))
        parts["activation"] = activation

        if return_normie:
            def call(x):
                normie = normalizer(x)
                return (normie, activation(normie))
        else:
            def call(x):
                y = normalizer(x)
                return activation(y)

    else:
        def call(x):
            return normalizer(x)
    return agent.pull_brick(parts)

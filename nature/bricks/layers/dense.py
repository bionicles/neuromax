import tensorflow as tf

from nature import get_l1_l2, get_chaos
from tools import make_uuid

K = tf.keras
L = K.layers

LAYER_CLASS = L.Dense  # NoiseDrop
UNITS = 256


def use_dense(
        agent, id, input,
        layer_class=LAYER_CLASS, units=UNITS, return_brick=False):
    id = make_uuid([id, "dense"])
    layer = layer_class(
            units,
            kernel_regularizer=get_l1_l2(),
            activity_regularizer=get_l1_l2(),
            bias_regularizer=get_l1_l2(),
            kernel_initializer=get_chaos(),
            bias_initializer=get_chaos(bias=True),
        )
    parts = dict(layer=layer)
    call = layer.call

    return agent.pull_brick(parts)

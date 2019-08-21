import tensorflow as tf

from nature import use_dense, swish
from tools import make_uuid
K = tf.keras

UNITS, FN, LAYERS = 256, swish, 2


def use_mlp(
        agent, id, out, layer_list=None, last_layer=None, return_brick=False):
    # we update id
    id = make_uuid([id, "mlp"])

    # we design layers if we didn't get them
    if layer_list is None:
        layer_list = [(UNITS, FN) for _ in range(LAYERS)]
        if last_layer is not None:
            layer_list[-1] = last_layer
    # we make parts
    layers = []
    for units, fn in layer_list:
        _, layer = use_dense(
            agent, id, out, units=units, fn=fn, return_brick=True)
        layers.append(layer)
    model = K.Sequential(layers)
    parts = dict(layers=layers, model=model)
    call = model
    return agent.pull_brick(id, parts, call, out, return_brick)

import tensorflow as tf

from nature import use_multiply, use_dense, use_input
from tools import make_uuid

K = tf.keras
L = K.layers


def use_flatten_resize_reshape(
        agent, id, input, out_spec=None, return_brick=False):
    id = make_uuid([id, "flatten_resize_reshape"])
    out_spec = out_spec if out_spec else agent.code_spec
    input_layer = use_input(agent, id, input)
    out = L.Flatten()(input_layer)
    if len(out.shape) is 3:
        out = use_multiply(agent, id, out_spec.size, out)
    else:
        out = use_dense(agent, id, units=out_spec.size)(out)
    out = L.Reshape(out_spec.shape)(out)
    model = K.Model(input, out, name=f"{id}_model")
    parts = dict(model=model)
    call = model.call
    return agent.build_brick(id, parts, call, input, return_brick)

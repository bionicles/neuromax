# interface.py - bion
# to destroy bugs, interface encapsulates autoencoders for common data formats
import tensorflow as tf

from nature import use_flatten_resize_reshape, use_norm_preact
from nature import use_fn, use_input

from tools import concat_coords, get_spec, log, make_uuid

K = tf.keras
L = K.layers

SIMPLE_INTERFACES = [
    'code_interface', 'onehot_actuator', 'discrete_actuator',
    'box_sensor', 'float_sensor', 'float_actuator']


def use_interface(
        agent, id, input=None,
        in_spec=None, out_spec=None, n=None, return_brick=False
        ):
    log("allow specs to be optional")
    out_spec = out_spec if out_spec else agent.code_spec
    in_spec = in_spec if in_spec else agent.code_spec
    log("trust input shape more than in_spec shape")
    if input:
        if in_spec.shape is not input.shape:
            in_spec = get_spec(**in_spec) if in_spec else get_spec(format="code")
            in_spec.shape = input.shape
    log("in_spec", in_spec)
    log("out_spec", out_spec)

    log("update the id")
    id = f"{id}{f'_{n}' if n else ''}"
    id = make_uuid([id, in_spec.format, "to", out_spec.format])

    log("decide what to build")
    return_normie = False
    if in_spec.format is not "code" and out_spec.format is "code":
        model_type = f"{in_spec.format}_sensor"
        return_normie = True
    if in_spec.format is "code" and out_spec.format is "code":
        model_type = f"code_interface"
    if in_spec.format is "code" and out_spec.format is not "code":
        model_type = f"{out_spec.format}_actuator"

    log("build the input")
    out = input_layer = use_input(agent, id, in_spec)
    parts = dict(input_layer=input_layer)

    log("resize images")
    if in_spec.format is "image":
        log("image, resizing")
        resizer = L.Lambda(lambda x: tf.image.resize(x, in_spec.shape))
        parts["resizer"] = resizer
        out = resizer(out)
        log("concat coordinates")
        parts["coordinator"] = coordinator = L.Lambda(concat_coords)
        out = coordinator(out)

    log("normalize inputs")
    out, norm_preact = use_norm_preact(
        agent, id, out, return_normie=return_normie, return_brick=True)
    parts["norm_preact"] = norm_preact

    log("unpack the normie for sensors")
    if "sensor" in model_type:
        normie, out = out

    log("create the output and interfacer ")
    if model_type in SIMPLE_INTERFACES:
        out, interfacer = use_flatten_resize_reshape(
            agent, id, out, in_spec, out_spec, return_brick=True)
    else:  # need a specialized interfacer
        out, interfacer = getattr("interfacer", f"use_{model_type}")(
            agent, id, input, in_spec, out_spec, return_brick=True)
    parts['interfacer'] = interfacer

    if out_spec.format in ["onehot", "discrete"]:
        log("using softmax on", id)
        out, fn = use_fn(agent, id, out, fn="softmax", return_brick=True)

    log("create the model and build the brick")
    outputs = (normie, out) if "sensor" in model_type else out
    parts["model"] = model = K.Model(input_layer, outputs)
    call = model.call
    return agent.pull_brick(parts)

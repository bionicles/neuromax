# interface.py - bion
# to destroy bugs, interface encapsulates autoencoders for common data formats
import tensorflow as tf

from tools import concat_coords, get_spec, log, make_id

K = tf.keras
L = K.layers

SIMPLE_INTERFACES = [
    'code_interface', 'onehot_actuator', 'discrete_actuator',
    'box_sensor', 'float_sensor', 'float_actuator']


def use_interface(agent, in_spec_or_tensor, out_spec_or_tensor=None):
    log("allow specs to be optional")
    if not out_spec_or_tensor:
        out_spec = agent.code_spec
    log("out_spec", out_spec)

    log("update the id")
    if 'n' in keys:
        n = parts['n']
        id = f"{id}_{n}"
    id = make_id(
        [id, "interface", str(in_spec.shape), "to", str(out_spec.shape)],
        reuse=reuse)

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
    input_parts = dict(brick_type="input", id=id, in_spec=in_spec)
    out = input_layer = agent.pull_brick(input_parts)
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

    log(f"normalize inputs... return_normie={return_normie}")
    norm_preact_parts = AttrDict(
        brick_type="norm_preact", id=id, inputs=out,
        return_normie=return_normie)
    norm_preact = agent.pull_brick(norm_preact_parts)
    parts["norm_preact"] = norm_preact

    log(f"unpack 'normie' from 'out' for sensors, out={out}")
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
    parts["model"] = K.Model(input_layer, outputs)
    parts["call"] = parts["model"].call
    return parts

import tensorflow as tf

from nature import use_flatten_resize_reshape, use_dense_block, use_input
from nature import use_deconv_2D, use_dense
from tools import make_uuid, log

K = tf.keras
L, B = K.layers, K.backend

BLOCK_FN = use_dense_block
LAST_DECODER_FN = "tanh"
N_BLOCKS = 2


def sample_fn(args):
    code_mean, code_variance = args
    epsilon = tf.random.normal(tf.shape(code_variance))
    sample = code_mean + B.exp(0.5 * code_variance) * epsilon
    return sample


def use_image_sensor(
        agent, id, input,
        out_spec=None, block_fn=BLOCK_FN, return_brick=False):
    id = make_uuid([id, "image_actuator"])
    log("handle specs")
    if out_spec is None:
        out_spec = agent.code_spec
    in_spec = agent.image_spec
    log("make model")
    input_layer = use_input(agent, id, in_spec)
    out = block_fn(agent, id, input)
    for block_number in range(N_BLOCKS - 1):
        out = block_fn(agent, id, out)
    out = L.Flatten()(out)
    code_mean = use_dense(agent, f"{id}_mean", out, units=out_spec.size)
    code_var = use_dense(agent, f"{id}_var", out, units=out_spec.size)
    sampled_tensor = L.Lambda(sample_fn, name='code')([code_mean, code_var])
    code = L.Reshape(out_spec.shape)(sampled_tensor)
    model = K.Model(input_layer, code)
    parts = dict(model=model, in_spec=in_spec, out_spec=out_spec)
    log("make call and brick")

    def call(x):
        x = tf.image.resize(x, in_spec.shape)
        return model(x)
    return agent.pull_brick(id, parts, call, input, return_brick)


def use_image_actuator(
        agent, id, input,
        out_spec=None, block_fn=BLOCK_FN, return_brick=False):
    id = make_uuid([id, "image_actuator"])

    if out_spec is None:
        out_spec = agent.image_spec

    input_layer = use_input(agent, id, input)
    out = use_flatten_resize_reshape(agent, id, input_layer, out_spec=out_spec)
    for layer_number in range(N_BLOCKS):
        out = use_dense_block(agent, id, out)  # dense for convex loss
    out = use_deconv_2D(
        agent, f"{id}_image_out_layer", out,
        filters=out_spec.shape[-1], fn=LAST_DECODER_FN, padding="same")
    model = K.Model(input_layer, out)
    parts = dict(model=model, out_spec=out_spec)
    call = model.call
    return agent.pull_brick(id, parts, call, input, return_brick)

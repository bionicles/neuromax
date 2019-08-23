import tensorflow as tf

from nature import use_linear, use_input
from tools import log

K = tf.keras
L, B = K.layers, K.backend


def sample_fn(args):
    code_mean, code_variance = args
    epsilon = tf.random.normal(tf.shape(code_variance))
    sample = code_mean + B.exp(0.5 * code_variance) * epsilon
    return sample


def use_image_sensor(agent, out_spec):
    log("handle image_sensor specs")
    out_spec = agent.code_spec
    in_spec = agent.image_spec

    log("make image_sensor layers")
    input_layer = input_layer = use_input(agent, in_spec)
    dense_block = agent.pull_brick("dense_block")
    code_mean_layer = use_linear(out_spec.size)
    code_var_layer = use_linear(out_spec.size)
    flatten = L.Flatten()
    sampler = L.Lambda(sample_fn, name='code')
    reshape = L.Reshape(out_spec.shape)

    log("use image_sensor layers")
    out = dense_block(input_layer)
    out = flatten(out)
    code_mean = code_mean_layer(out_spec.size)
    code_var = code_var_layer(out_spec.size)
    sample = sampler([code_mean, code_var])
    code = reshape(sample)

    log("make model, call, and return call")
    model = K.Model(input_layer, code)

    def call(self, x):
        x = tf.image.resize(x, in_spec.shape)
        return model(x)
    return call

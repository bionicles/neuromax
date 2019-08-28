import tensorflow as tf

from nature import use_dense_block, use_linear, use_fn

K = tf.keras
L, B = K.layers, K.backend


def sample_fn(args):
    mean, variance = args
    epsilon = tf.random.truncated_normal(tf.shape(variance))
    return mean + B.exp(0.5 * variance) * epsilon


def use_image_sensor(agent):
    out_spec = agent.code_spec
    dense_block = use_dense_block()
    flatten = L.Flatten()
    mean_layer = use_linear(units=out_spec.size)
    var_layer = use_linear(units=out_spec.size)
    softplus = use_fn("softplus")
    sampler = L.Lambda(sample_fn, name='sample')
    reshape = L.Reshape(out_spec.shape)

    def call(x):
        x = dense_block(x)
        x = flatten(x)
        mean = mean_layer(x)
        var = var_layer(x)
        var = softplus(var)
        sample = sampler([mean, var])
        return reshape(sample)
    return call

import tensorflow as tf

from nature import DenseBlock, Linear, Fn, Brick

K = tf.keras
L, B = K.layers, K.backend


def sample_fn(args):
    mean, variance = args
    epsilon = tf.random.truncated_normal(tf.shape(variance))
    return mean + B.exp(0.5 * variance) * epsilon


def ImageSensor(agent):
    out_spec = agent.code_spec
    dense_block = DenseBlock()
    flatten = L.Flatten()
    mean_layer = Linear(units=out_spec.size)
    var_layer = Linear(units=out_spec.size)
    softplus = Fn(key="softplus")
    sampler = L.Lambda(sample_fn, name='sample')
    reshape = L.Reshape(out_spec.shape)

    def call(self, x):
        x = dense_block(x)
        x = flatten(x)
        mean = mean_layer(x)
        var = var_layer(x)
        var = softplus(var)
        sample = sampler([mean, var])
        return reshape(sample)
    return Brick(
        out_spec, dense_block, flatten, mean_layer, var_layer,
        softplus, sampler, reshape, call, agent)

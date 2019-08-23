import tensorflow as tf

KI = tf.keras.initializers

KERNELS = dict(swish=2.952, relu=2, tanh=1.722)
BIASES = dict(swish=0.04, relu=0, tanh=0.04)

DISTRIBUTION = "uniform"
DEFAULT_BIAS = False
DEFAULT_FN = "relu"
MODE = 'fan_avg'

PHI = 1.618
OFFSET = 0.01


def get_phi(shape):
    return KI.TruncatedNormal(mean=0., stddev=PHI)


def get_chaos(bias=DEFAULT_BIAS, fn=DEFAULT_FN):
    return EdgeOfChaos(bias, fn)


class EdgeOfChaos(KI.VarianceScaling):
    """The edge of chaos truncated normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.
    Args:
        kernel: boolean flag, if we're initializing a kernel, else do a bias
        fn_name: string name of the activation function
    References:
      [Glorot et al., 2010]
      http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
      [Hayou et al., 2010]
      https://arxiv.org/pdf/1805.08266.pdf
    """

    def __init__(self, bias, fn_name, seed=None):
        if 'relu' in fn_name:
            fn_name = 'relu'
        dictionary = BIASES if bias else KERNELS
        super(EdgeOfChaos, self).__init__(
            scale=dictionary[fn_name],
            mode="fan_avg",
            distribution=DISTRIBUTION,
            seed=seed)

    def get_config(self):
        return {"seed": self.seed}

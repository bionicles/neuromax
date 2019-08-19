import tensorflow as tf

KERNELS = dict(swish=2.952, relu=2, tanh=1.722)
BIASES = dict(swish=0.04, relu=0, tanh=0.04)


class EdgeOfChaos(tf.keras.initializers.VarianceScaling):
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

    def __init__(self, kernel, fn_name, seed=None):
        dictionary = KERNELS if kernel else BIASES
        super(EdgeOfChaos, self).__init__(
            scale=dictionary[fn_name],
            mode="fan_avg",
            distribution="truncated_normal",
            seed=seed)

    def get_config(self):
        return {"seed": self.seed}

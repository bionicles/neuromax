import tensorflow as tf

from nature import Init, L1L2

ACTIVATION = None
INITIALIZER = Init
REGULARIZER = L1L2
STDDEV = 0.04
P_DROP = 0.5
UNITS = 16


def NoiseDrop(
        kernel_regularizer=REGULARIZER,
        activity_regularizer=REGULARIZER,
        bias_regularizer=REGULARIZER,
        kernel_initializer=INITIALIZER,
        bias_initializer=INITIALIZER,
        activation=ACTIVATION,
        units=UNITS,
        **kwargs
        ):
    return _NoiseDrop(
            units=units,
            activation=activation,
            kernel_regularizer=kernel_regularizer(),
            activity_regularizer=activity_regularizer(),
            bias_regularizer=bias_regularizer(),
            kernel_initializer=kernel_initializer(),
            bias_initializer=bias_initializer(dist='truncated'),
            **kwargs
            )

class _NoiseDrop(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super(_NoiseDrop, self).__init__(*args, **kwargs)

    @tf.function
    def add_noise(self):
        return (
            self.kernel + tf.random.truncated_normal(
                tf.shape(self.kernel), stddev=STDDEV),
            self.bias + tf.random.truncated_normal(
                tf.shape(self.bias), stddev=STDDEV))

    @tf.function
    def call(self, x):
        kernel, bias = self.add_noise()
        return self.activation(
                    tf.nn.bias_add(
                        tf.keras.backend.dot(x, tf.nn.dropout(kernel, P_DROP)),
                        bias))

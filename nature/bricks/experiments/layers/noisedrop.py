import tensorflow as tf


STDDEV = 0.005
P_DROP = 0.5


class NoiseDrop(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        super(NoiseDrop, self).__init__(*args, **kwargs)

    @tf.function
    def add_noise(self):
        return (
            self.kernel + tf.random.truncated_normal(
                tf.shape(self.kernel), stddev=STDDEV),
            self.bias + tf.random.truncated_normal(
                tf.shape(self.bias), stddev=STDDEV))

    def call(self, x):
        kernel, bias = self.add_noise()
        return self.activation(
                    tf.nn.bias_add(
                        tf.keras.backend.dot(x, tf.nn.dropout(kernel, P_DROP)),
                        bias))

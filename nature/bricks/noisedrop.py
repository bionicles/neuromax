import tensorflow as tf


MIN_STDDEV, MAX_STDDEV = 1e-4, 0.1
P_DROP = 0.5


class NoiseDrop(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        self.stddev = kwargs.pop('stddev')
        super(NoiseDrop, self).__init__(*args, **kwargs)

    @tf.function
    def add_noise(self):
        return (
            self.kernel + tf.random.truncated_normal(
                tf.shape(self.kernel), stddev=self.stddev),
            self.bias + tf.random.truncated_normal(
                tf.shape(self.bias), stddev=self.stddev))

    def call(self, x):
        kernel, bias = self.add_noise()
        return self.activation(
                    tf.nn.bias_add(
                        tf.keras.backend.dot(x, tf.nn.dropout(kernel, P_DROP)),
                        bias))

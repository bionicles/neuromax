import tensorflow as tf
K, L = tf.keras, tf.keras.layers


class LogisticMap(L.Layer):

    def __init__(self):
        super(LogisticMap, self).__init__()
        self.r = 3.57
        # self.r = tf.random.uniform((), minval=3.57, maxval=4.)
        self.built = True

    @tf.function
    def call(self, x):
        min = tf.math.reduce_min(x)
        x = (x - min) / (tf.math.reduce_max(x) - min)
        return self.r * x * (1. - x)

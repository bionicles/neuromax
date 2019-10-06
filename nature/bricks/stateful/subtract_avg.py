import tensorflow as tf
from nature import RunningAvg


class SubtractAvg(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def build(self, shape):
        self.avg = RunningAvg()
        super().build(shape)

    @tf.function
    def call(self, x):
        return x - self.avg(x)

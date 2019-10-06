import tensorflow as tf


class RunningAvg(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def build(self, shape):
        self.avg = self.add_weight("avg", shape)
        self.step = 0
        super().build(shape)

    @tf.function
    def call(self, x):
        avg = (x + self.step * self.avg) / (self.step + 1)
        self.step = self.step + 1
        self.avg.assign(avg)
        return avg

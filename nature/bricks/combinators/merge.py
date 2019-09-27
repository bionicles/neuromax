import tensorflow as tf
import nature
L = tf.keras.layers
AXIS = 1


class Merge(L.Layer):

    def __init__(self, code_shape, axis=AXIS):
        super(Merge, self).__init__()
        self.code_shape = code_shape
        self.d_code = code_shape[-1]
        self.axis = axis

    def build(self, shapes):
        self.expander = L.Lambda(lambda x: tf.expand_dims(x, -1))
        self.changer = nature.Conv1D(units=self.d_code - 1)
        self.merge = L.Concatenate(self.axis)
        super().build(shapes)

    @tf.function
    def call(self, x):
        items = []
        for item in x:
            if len(tf.shape(item)) is 2:
                item = self.expander(item)
                item = tf.concat([item, self.changer(item)], -1)
            items.append(item)
        y = self.merge(items)
        return y

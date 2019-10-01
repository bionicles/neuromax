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
        self.interfaces = []
        for shape in shapes:
            expander, changer = tf.identity, tf.identity
            if len(shape) is 2:
                expander = self.expander
                shape = list(shape) + [1]
            if shape[-1] is not self.d_code:
                if shape[-1] < self.d_code:
                    changer = L.Lambda(
                        lambda x: tf.concat([x, self.changer(x)], -1))
                else:
                    changer = nature.Conv1D(units=self.d_code)
            self.interfaces.append((changer, expander))
        super().build(shapes)

    @tf.function
    def call(self, x):
        items = []
        for k, item in enumerate(x):
            changer, expander = self.interfaces[k]
            items.append(changer(expander(item)))
        y = self.merge(items)
        return y

import tensorflow as tf
import nature

L = tf.keras.layers
LAYER_FN = nature.Layer
N_LOOPS = 2


@tf.function
def ERROR(true, pred):
    return tf.math.abs(tf.math.subtract(true, pred))


class Recirculator(L.Layer):

    def __init__(self, units=None, layer_fn=LAYER_FN):
        super(Recirculator, self).__init__()
        self.layer_fn = layer_fn

    def build(self, shape):
        self.code = self.add_weight("code", shape)
        self.encode = self.layer_fn(units=shape[-1])
        self.decode = self.layer_fn(units=shape[-1])
        self.fn = nature.Fn("tanh")
        self.out = L.Add()
        super().build(shape)

    # @tf.function
    def call(self, x):
        prev_code = tf.identity(self.code)
        prev_state = x
        errors = []
        for n in range(N_LOOPS):
            prediction = self.fn(self.decode(prev_code))
            code = self.fn(self.encode(prev_state))
            reconstruction = self.fn(self.decode(code))
            e1 = ERROR(prev_state, prediction)
            e2 = ERROR(prev_state, reconstruction)
            e3 = ERROR(x, reconstruction)
            e4 = ERROR(prev_code, code)
            self.add_loss([e1, e2, e3, e4])
            errors.extend([e1, e2, e3, e4])
            prev_state = reconstruction
            prev_code = code
        y = self.out(errors)
        return y

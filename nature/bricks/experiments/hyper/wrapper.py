from functools import reduce
from operator import mul
import tensorflow as tf
from tools import log
B = tf.keras.backend
L = tf.keras.layers


def get_size(shape):
    return shape[0] if len(shape) is 1 else reduce(mul, shape)


def pipe(*args, repeats=1):

    def call(x):
        for i in range(repeats):
            for arg in args:
                x = arg(x)
        return x
    return call


class Hyper(L.Wrapper):

    def __init__(self, AI, layer):
        super().__init__(layer)

    def build(self, shape):
        super().build(shape)
        shapes = [w.shape for w in self.layer.get_weights()]
        self.hypers = []
        for k, shape in enumerate(shapes):
            log("shape", k, shape)
            hyper = L.Dense(get_size(shape))
            super().__setattr__("hyper_{k}", hyper)
            reshape = L.Reshape(shape)
            super().__setattr__("reshape_{k}", reshape)
            self.hypers.append(pipe(hyper, reshape))
        self.flatten = L.Flatten()

    @tf.function
    def call(self, inputs):
        batch = B.int_shape(inputs)[0]
        Y = []
        for x in tf.split(inputs, batch):
            x = self.flatten(x)
            weights = []
            for hyper in self.hypers:
                w = hyper(x)
                w = tf.squeeze(w, 0)
                weights.append(w)
            self.layer.set_weights(weights)
            y = self.layer(x)
            Y.append(y)
        return tf.concat(Y, 0)

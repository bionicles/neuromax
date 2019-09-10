import tensorflow as tf

from nature import Norm, Fn, Resizer

L = tf.keras.layers


class Classifier(L.Layer):

    def __init__(self, spec):
        super(Classifier, self).__init__()
        self.norm = Norm()
        self.resizer = Resizer(spec.shape)
        self.softmax = Fn(key='softmax')
        self.built = True

    @tf.function
    def call(self, x):
        x = self.resizer(self.norm(x))
        x = self.softmax(x)
        return x

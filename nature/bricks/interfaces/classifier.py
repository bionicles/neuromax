import tensorflow as tf

from nature import Norm, Fn, Linear, Resizer

L = tf.keras.layers


class Classifier(L.Layer):

    def __init__(self, spec):
        super(Classifier, self).__init__()
        self.linear1 = Linear()
        self.norm1 = Norm()
        self.norm2 = Norm()
        self.norm3 = Norm()
        self.resizer = Resizer(spec.shape)
        self.softmax = Fn(key='softmax')

    @tf.function
    def call(self, x):
        x = self.linear1(self.norm1(x))
        x = self.resizer(self.norm2(x))
        x = self.softmax(self.norm3(x))
        return x

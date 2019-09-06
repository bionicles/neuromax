import tensorflow as tf

from nature import Norm, Fn, Linear, Resizer

L = tf.keras.layers


class Classifier(L.Layer):

    def __init__(self, spec):
        super(Classifier, self).__init__()
        self.m1 = Linear()
        self.m2 = Linear()
        self.n1 = Norm()
        self.n2 = Norm()
        self.n3 = Norm()
        self.n4 = Norm()
        self.resizer = Resizer(spec.shape)
        if spec.format is "discrete":
            self.classifier = Fn(key='soft_argmax')
        elif spec.format is 'onehot':
            self.classifier = Fn(key='softmax')

    def call(self, x):
        x = self.m1(self.n1(x))
        x = self.m2(self.n2(x))
        x = self.resizer(self.n3(x))
        return self.classifier(self.n4(x))

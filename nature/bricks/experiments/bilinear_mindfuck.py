import tensorflow as tf

L = tf.keras.layers
UNITS = 16
N = 5

def mindfuck(x, n=N, units=UNITS):
    for _ in range(n):
        x = L.LayerNormalization(Bilinear()([x, L.Attention([
                L.Dense(units)(x), L.Dense(units)(x), L.Dense(units)(x)])]))
    return x

def hard_mindfuck(x, n=N, units=None):
    for _ in range(n):
        x = L.LayerNormalization(x + L.Attention([x, x, x]))
    return x

def hard_bayesian_mindfuck(x, n=N, units=None):
    for _ in range(n):
        x = L.LayerNormalization(L.Multiply()([x, L.Attention([x, x, x])]))
    return x

def hard_bayesian_mindfuck(x, n=N, units=None):
    for _ in range(n):
        x = L.LayerNormalization(L.Multiply()([x, L.Attention([
                L.Dense(units)(x), L.Dense(units)(x), L.Dense(units)(x)])]))
    return x


class Bilinear(L.Layer):

    def __init__(self, units=UNITS):
        super().__init__()

    def build(self, shapes):
        size1 = tf.math.reduce_prod(shapes[0][1:])
        size2 = tf.math.reduce_prod(shapes[1][1:])
        self.A = self.add_weight("kernel", [size1, size2, self.units])
        self.B = self.add_weight("bias", [self.units])
        super().build(shapes)

    @tf.function
    def call(self, ab):
        return B.batch_dot(B.batch_dot(x[0], self.A), x[1]) + self.B

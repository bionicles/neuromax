import tensorflow as tf
from tools import make_id
L = tf.keras.layers


class AOA(L.Layer):

    def __init__(self, AI, units=UNITS):
        layers = AI.pull("AOA_layers", ["monolayer", "bilayer"])
        super().__init__(name=make_id(f"{layers}_attn"))
        self.call = self.call_monolayer
        if bilayer:
            self.call = self.call_bilayer
            self.q2 = L.Dense(units)
            self.k2 = L.Dense(units)
            self.v2 = L.Dense(units)
        self.attn = L.Attention()
        self.q1 = L.Dense(units)
        self.k1 = L.Dense(units)
        self.v1 = L.Dense(units)
        self.built = True

    @tf.function
    def call_monolayer(self, x):
        a = self.attn([self.q1(x), self.k1(x), self.v1(x)])
        return self.attn([a, a, a])

    @tf.function
    def call_monolayer(self, x):
        a = self.attn([self.q1(x), self.k1(x), self.v1(x)])
        return self.attn([self.q2(a), self.k2(a), self.v2(a)])

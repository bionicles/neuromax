import tensorflow as tf

L = tf.keras.layers


class DNC_RNN:
    """An RNN with DNC cells"""

    def __init__(self, dnc_cell):
        self.dnc_cell = dnc_cell
        # TODO: make a DNC RNN

    def call(self, input):
        return L.RNN(self.dnc_cell)
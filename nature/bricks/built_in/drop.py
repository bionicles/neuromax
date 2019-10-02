import tensorflow as tf

L = tf.keras.layers

RATE = 0.5


def Drop(rate=RATE):
    return L.Dropout(rate)

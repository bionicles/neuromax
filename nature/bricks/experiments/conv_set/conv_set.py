import tensorflow as tf

L = tf.keras.layers

N = 3

class ConvSet1D(L.Layer):

    def __init__(self, n=N):
        

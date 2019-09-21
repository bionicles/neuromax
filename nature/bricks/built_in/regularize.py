import tensorflow as tf

K, L = tf.keras, tf.keras.layers

L1, L2 = 0.000, 0.000


def L1L2(l1=L1, l2=L2):
    return K.regularizers.L1L2(l1=l1, l2=l2)

def Regularizer(l1=L1, l2=L2):
    return L.ActivityRegularization(l1=l1, l2=l2)

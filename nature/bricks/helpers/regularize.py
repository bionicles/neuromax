import tensorflow as tf

K, L = tf.keras, tf.keras.layers

L1, L2 = 0.0001, 0.0001


def L1L2(l1=L1, l2=L2):
    return tf.keras.regularizers.l1_l2(l1=l1, l2=l2)


def Regularizer(l1=L1, l2=L2):
    return L.ActivityRegularization(l1=l1, l2=l2)

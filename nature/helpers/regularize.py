import tensorflow as tf

K = tf.keras

L1, L2 = 0.001, 0.0001


def L1L2():
    return K.regularizers.L1L2(l1=L1, l2=L2)

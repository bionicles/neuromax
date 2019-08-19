import tensorflow as tf
K = tf.keras

L1, L2 = 1e-4, 1e-4


def get_l1_l2():
    return K.regularizers.L1L2(l1=L1, l2=L2)


def get_l1():
    return K.regularizers.L1(l1=L1)


def get_l2():
    return K.regularizers.L2(l2=L2)

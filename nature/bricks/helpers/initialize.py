# https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404
import tensorflow as tf


from nature import EdgeOfChaos
initializers = tf.keras.initializers


DIST = "truncated"


def Init(dist=DIST):
    if dist is "chaos":
        return EdgeOfChaos()
    elif dist is "truncated":
        return initializers.TruncatedNormal()
    elif dist is "he":
        return initializers.he_normal()
    else:
        return initializers.glorot_normal()

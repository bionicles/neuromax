# https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404
import tensorflow as tf

initializers = tf.keras.initializers

DIST = "glorot"
PHI = 1.61803398875
MEAN = 0.


def Init(dist=DIST):
    if dist is "truncated":
        return initializers.TruncatedNormal(mean=MEAN, stddev=PHI)
    elif dist is "he_uniform":
        return initializers.he_uniform()
    else:
        return initializers.glorot_uniform()
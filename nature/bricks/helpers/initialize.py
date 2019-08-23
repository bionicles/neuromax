# https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404
import tensorflow as tf

initializers = tf.keras.initializers


def get_init(he=False):
    """get glorot_uniform unless kwarg he=True"""
    if he is True:
        return initializers.he_uniform()
    else:
        return initializers.glorot_uniform()

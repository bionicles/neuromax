import tensorflow as tf

K = tf.keras
L = K.layers

LAYER_FN = L.Dense
KWARG = 1


def get_xxx_out(
        agent, id, input, layer_fn=LAYER_FN, kwarg=KWARG):
    return layer_fn()(input)

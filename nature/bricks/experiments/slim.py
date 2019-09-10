import tensorflow as tf

from nature import Norm, Fn, FC

L = tf.keras.layers

LAYER = FC


class Slim(L.Layer):

    def __init__(self):
        super(Slim, self).__init__()

    def build(self, shape):
        d_in = shape[-1]
        for block_number in range(d_in):
            norm = Norm()
            fn = Fn()
            layer = Layer(units_or_filters=1, layer_fn=LAYER)
            concat = L.Add()
            block = (norm, fn, layer, concat)
            setattr(self, f"norm_{block_number}", norm)
            setattr(self, f"fn_{block_number}", fn)
            setattr(self, f"layer_{block_number}", layer)
            setattr(self, f"concat_{block_number}", concat)

    @tf.function
    def call(self, x):
        norm, fn, layer, _ = blocks[0]
        x = layer(fn(norm(x)))
        for norm, fn, layer, concat in blocks[1:]:
            y = layer(fn(norm(x)))
            x = concat([x, y])
        return x

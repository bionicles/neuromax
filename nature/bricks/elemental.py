import tensorflow as tf

from nature import use_linear

K = tf.keras
L = K.layers

UNITS = 128


def use_multiply(units=UNITS):
    linear1 = use_linear(units)
    linear2 = use_linear(units)
    multiplier = L.Multiply()

    def call(x):
        o1 = linear1(x)
        o2 = linear2(x)
        return multiplier([o1, o2])
    return call


# commenting out this guy because he's likely to hit shape bugs
# def use_add(units, tensor):
#     linear = use_linear(units)
#     adder = L.Add()
#
#     def call(x):
#         y = linear(x)
#         y = adder([x, y])
#         return y
#     return call

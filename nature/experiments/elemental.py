import tensorflow as tf

from nature import Linear, Brick

K = tf.keras
L = K.layers

UNITS = 128


def Multiply(agent, units=UNITS):
    linear1 = Linear(units)
    linear2 = Linear(units)
    multiplier = L.Multiply()

    def call(self, x):
        o1 = linear1(x)
        o2 = linear2(x)
        return multiplier([o1, o2])
    return Brick(linear1, linear2, multiplier, call, agent)


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

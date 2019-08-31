import tensorflow as tf

from nature import Brick, Linear

K = tf.keras
L, BE = K.layers, K.backend

UNITS = 4


def Quadratic(agent, units=UNITS):
    R = Linear(units=units)
    G = Linear(units=units)
    B = Linear(units=units)
    square = L.Lambda(BE.square)
    entrywise_multiply = L.Multiply()

    def call(self, x):
        return entrywise_multiply([R(x), G(x)]) + B(square(x))
    return Brick(R, G, B, square, entrywise_multiply, call, agent=agent)

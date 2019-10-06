import tensorflow as tf
import nature

MIN, MAX = 2, 4
UNITS = 3

L = tf.keras.layers


def Chain(AI, units=UNITS):
    n_bricks = AI.pull("chain_length", MIN, MAX)

    def call(x):
        arr = []
        y = x
        for n in range(n_bricks):
            brick = nature.Brick(f"chain_brick_{n}", AI, no_combinators=1)
            y = brick(AI, units=units)(y)
            arr.append(y)
        return L.Concatenate(1)(arr)
    return call

import tensorflow as tf

from nature import Norm, Fn, Linear, Brick

L = tf.keras.layers

N_BLOCKS = 9


def use_slim(agent, n_blocks=N_BLOCKS):
    blocks = [
        (Norm(), Fn(), Linear(units=1), L.Add())
        for _ in range(N_BLOCKS)
        ]

    def call(x):
        for norm, fn, linear, add in blocks:
            y = linear(fn(norm(x)))
            x = add([x, y])
        return x
    return Brick(blocks, call, agent=agent)

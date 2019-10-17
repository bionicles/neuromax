import nature

MIN, MAX = 2, 8
UNITS = 4


def Stack(AI, units=UNITS):
    n_bricks = AI.pull("stack_size", MIN, MAX)

    def call(x):
        for n in range(n_bricks):
            brick = nature.Brick(f"stack_brick_{n}", AI, no_combinators=1)
            x = brick(AI, units=units)(x)
        return x
    return call

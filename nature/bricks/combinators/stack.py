import nature

MIN, MAX = 2, 4
UNITS = 4


def Stack(AI, units=UNITS):
    n_bricks = AI.pull("stack_size", MIN, MAX, id=False)

    def call(x):
        for n in range(n_bricks):
            brick = nature.Brick(
                f"stack_brick_{n}", AI, no_combinators=True)
            x = brick(AI, units=units)(x)
        return x
    return call

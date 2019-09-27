from tools import pipe, stack, safe_sample
from random import choice
import nature

MIN, MAX = 1, 1
UNITS = 3

OPTIONS = [nature.Transformer]


def Stack(units=UNITS, options=OPTIONS):
    if isinstance(options, list):
        constructor = choice(options)
    n_bricks = safe_sample(MIN, MAX)
    def wrapped():
        return constructor(units=units)
    return pipe(*stack(wrapped, n_bricks))

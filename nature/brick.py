from tools import pipe, stack, safe_sample
from random import choice
import nature

BRICK_OPTIONS = [
    (nature.ResBlock, 1, 4),
    (nature.DenseBlock, 1, 4),
    (nature.Quadratic, 1, 4),
    (nature.FC, 1, 2),
    (nature.Conv1D, 1, 2),
    (nature.Predictor, 1, 4),
    (nature.AllAttention, 1, 4)
]


def Brick(code_shape):
    constructor, min, max = choice(BRICK_OPTIONS)
    n_bricks = safe_sample(min, max)
    wrapped = constructor
    if constructor is nature.Predictor:
        def wrapped():
            return constructor(out_shape=code_shape)
    return pipe(*stack(wrapped, n_bricks))

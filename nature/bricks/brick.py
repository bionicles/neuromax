import nature
import random

bricks = [
    nature.Integral,
    nature.Recirculator,
    nature.Slim,
    nature.SWAG,
    nature.OP_1D,
    nature.MLP,
    nature.Transformer,
    nature.Derivative,
]


def pick_brick():
    return random.choice([nature.Chain, nature.Stack])


def Brick(id):
    return bricks[id] if id <= len(bricks) else pick_brick

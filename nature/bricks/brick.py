import nature
import random

bricks = [
    nature.Recirculator,
    nature.Slim,
    nature.SWAG,
    nature.OP_1D,
    nature.MLP,
    nature.Transformer,
    # nature.WideDeep,
]


def pick_brick():
    return random.choice([nature.Chain, nature.Stack])


def Brick(id):
    return bricks[id] if id <= len(bricks) else pick_brick

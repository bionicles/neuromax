import tensorflow as tf
import nature
import random

bricks = [
    nature.Transformer,
    nature.Integral,
    nature.Delta,
    nature.Slim,
    nature.SWAG,
    nature.MLP,
    # nature.Recirculator,
    # nature.Derivative,
]


def pick_brick():
    return random.choice([nature.Chain, nature.Stack])


def Brick(id):
    return bricks[id] if id <= len(bricks) else pick_brick

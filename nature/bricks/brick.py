import nature

OPTIONS = [
    nature.Echo,
    nature.Circulator,
    nature.ConvSet,
    nature.MLP,
    nature.Attention,
    nature.SWAG,
    nature.Delta,
    nature.Transformer,
    nature.Slim,
]

COMBINATORS = [
    nature.Chain,
    nature.Stack
]


def Brick(id, AI, no_combinators=False):
    if no_combinators:
        brick = nature.Chain
        while brick in [nature.Chain, nature.Stack]:
            brick = AI.pull(f"brick_{id}_no_combinator", OPTIONS)
    else:
        brick = AI.pull(f"brick_{id}", OPTIONS + COMBINATORS)
    return brick

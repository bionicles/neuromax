import nature

OPTIONS = [
    nature.Circulator,
    nature.ConvSet,
    nature.MLP,
    nature.Attention,
    nature.SWAG,
    nature.Delta,
    nature.Transformer,
    nature.Slim,
    nature.Chain,
    nature.Stack,
    ]


def Brick(id, AI, no_combinators=False):
    if no_combinators:
        brick = nature.Chain
        while brick in [nature.Chain, nature.Stack]:
            brick = AI.pull(f"brick_{id}", OPTIONS)
    else:
        brick = AI.pull(f"brick_{id}", OPTIONS, id=False)
    return brick

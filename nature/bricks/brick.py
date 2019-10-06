import nature

OPTIONS = [
    'Echo',
    'Circulator',
    'ConvSet',
    'MLP',
    'Attention',
    'SWAG',
    'Delta',
    'Transformer',
    'Slim']

COMBINATORS = ['Chain', 'Stack']

ALL = OPTIONS + COMBINATORS


def Brick(id, AI, no_combinators=False):
    if no_combinators:
        key = AI.pull(f"brick_{id}_no_combinator", OPTIONS)
    else:
        key = AI.pull(f"brick_{id}", ALL)
    return getattr(nature, key)

from nanoid import generator
import importlib


class Brick:
    """Wrap ops inside an agent
    Args:
        agent: Agent instance for this brick
        brick_type: string "ClassName" of the desired brick
    """

    def __init__(self, *args, **kwargs):
        self.brick_type = kwargs["brick_type"]
        self.agent = kwargs["agent"]
        self.name = f"{self.brick_type}-{generator()}"
        Op = getattr(importlib.import_module("nature.bricks"), self.brick_type)
        self.op = Op(*args, **kwargs)
        self.__call__ = self.op.__call_

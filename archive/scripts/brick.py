import importlib


def get_brick(*args, **kwargs):
    """returns a brick
    kwargs: agent, brick_type
    """
    assert "brick_type" in kwargs.keys()
    assert "agent" in kwargs.keys()
    brick_type = kwargs["brick_type"]
    module_path = f"nature.bricks.{brick_type.lower()}"
    module = importlib.import_module(module_path)
    OpClass = getattr(module, brick_type)
    return OpClass(*args, **kwargs)


# class Brick:
#     """
#     Standard way to build bricks
#
#     kwargs:
#         agent: Agent instance for this brick
#         brick_type: string "ClassName" of the desired brick
#     """
#
#     def __init__(self, *args, **kwargs):
#         brick_type = kwargs["brick_type"]
#         module_path = f"nature.bricks.{brick_type.lower()}"
#         module = importlib.import_module(module_path)
#         OpClass = getattr(module, brick_type)
#         self.op = OpClass(*args, **kwargs)
#         self.__call__ = self.op.__call_

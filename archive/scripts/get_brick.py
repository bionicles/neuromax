import importlib


def get_brick(*args, **kwargs):
    brick_type = kwargs["brick_type"]
    module_path = f"nature.bricks.{brick_type.lower()}"
    module = importlib.import_module(module_path)
    OpClass = getattr(module, brick_type)
    return OpClass(*args, **kwargs)

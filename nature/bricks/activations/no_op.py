from nature import Fn


def NoOp(AI, *args, **kwargs):
    return Fn(AI, key="identity")

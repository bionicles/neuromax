def wrap_fn(fn, *args, **kwargs):
    def _fn(x):
        return fn(x, *args, **kwargs)
    return _fn

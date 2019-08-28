from nature import use_norm, use_fn


def use_norm_preact(key=None):
    norm = use_norm()
    fn = use_fn(key=key) if key else use_fn()

    def call(x):
        return fn(norm(x))
    return call

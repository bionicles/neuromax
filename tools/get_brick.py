# https://stackoverflow.com/a/45102583/4898464


def get_brick(brick_fn, out, return_brick):
    if out is None:
        return brick_fn
    out = brick_fn(out)
    return out, brick_fn if return_brick else out

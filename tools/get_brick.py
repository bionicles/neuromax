# https://stackoverflow.com/a/45102583/4898464


def get_brick(parts, fn, out, return_brick):
    if out is None:
        return fn
    out = fn(out)
    return out, fn if return_brick else out

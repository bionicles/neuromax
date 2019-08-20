

def build_brick(self, id, parts, call, out, return_brick):
    """construct a brick from parts and a function to use them

    https://stackoverflow.com/a/45102583/4898464
    """
    self.graph[id].brick = brick = Brick(self, id, parts, call, out)
    if out is None:
        return brick
    self.graph[id].out = out = brick(out)
    return out, brick if return_brick else out

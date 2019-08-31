from nature import Norm, Fn, Brick


def NormPreact(agent, key=None):
    norm = Norm()
    fn = Fn(key=key) if key else Fn()

    def call(self, x):
        return fn(norm(x))
    return Brick(norm, fn, call, agent)

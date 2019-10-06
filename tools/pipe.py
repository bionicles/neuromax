def pipe(*args, repeats=1):

    def call(x):
        for i in range(repeats):
            for arg in args:
                x = arg(x)
        return x
    return call

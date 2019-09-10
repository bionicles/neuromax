def stack(layer, repeats):
    return [layer() for _ in range(repeats)]

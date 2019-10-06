from nanoid import generate

NANOID_SIZE = 4


def make_id(name):
    return f"{name}_{generate('0123456789', NANOID_SIZE)}"

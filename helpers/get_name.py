from nanoid import generate

NANOID_SIZE = 16


def get_name(brick_type):
    return f"{brick_type}_{generate(size=NANOID_SIZE)}"

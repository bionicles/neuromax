from nanoid import generate

NANOID_SIZE = 4


def make_id(parts):
    id = f"{parts.id}_{parts.brick_type}"
    if "reuse" in parts.keys():
        if parts.reuse:
            return id
    else:
        return f"{id}_{generate('0123456789', NANOID_SIZE)}"

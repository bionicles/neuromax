from nanoid import generate

NANOID_SIZE = 4


def make_uuid(string_or_list):
    if isinstance(string_or_list, str):
        name = string_or_list
    elif isinstance(string_or_list, list):
        no_nones = filter(lambda x: x is not None, string_or_list)
        name = "_".join([str(x).lower() for x in no_nones])
    else:
        name = "bob"
    return f"{name}_{generate('0123456789', NANOID_SIZE)}"

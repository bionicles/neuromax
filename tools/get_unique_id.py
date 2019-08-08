from nanoid import generate

NANOID_SIZE = 16


def get_unique_id(string_or_list):
    if isinstance(string_or_list, str):
        name = string_or_list
    elif isinstance(string_or_list, list):
        name = string_or_list.join("_")
    else:
        name = "bob"
    return f"{name}_{generate(size=NANOID_SIZE)}"

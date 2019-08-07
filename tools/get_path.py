import os


def get_path(folder, name, extension):
    return os.path.join(folder, f"{name}.{extension}")

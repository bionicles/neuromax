# debug.py - bion
# why?: organize helper functions for debugging
from blessings import Terminal

T = Terminal()

DEBUG = True


def log(*args, color="white", debug=DEBUG):
    args = str(args)
    for char in ["(", ")", ",", "'"]:
        args = args.replace(char, '')
    if debug:
        print(getattr(T, color)(args))

# debug.py - bion
# why?: organize helper functions for debugging
from blessings import Terminal

T = Terminal()

DEBUG = True
color = "red"


def log(*args, debug=DEBUG):
    args = str(args)
    for char in ["(", ")", ",", "'"]:
        args = args.replace(char, '')
    if debug:
        print(getattr(T, color)(args))

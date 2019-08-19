# debug.py - bion
# why?: organize helper functions for debugging
from blessings import Terminal
from time import sleep

T = Terminal()

DEBUG = False

# color
# black
# red
# green
# yellow
# blue
# magenta
# cyan
# white


def log(*args, color="white", delay=0, debug=DEBUG):
    """
    Print args to the terminal

    Kwargs:
        color: string black, red, green, yellow, blue, magenta, cyan, white
        delay: number of seconds to pause after printing
    """
    args = str(args)
    for char in ["(", ")", ",", "'"]:
        args = args.replace(char, '')
    if debug:
        print(getattr(T, color)(args))
    if delay:
        sleep(delay)

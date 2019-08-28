# log.py - bion
# why?: log is shorter to type than 'print' and we can add extra features
from blessings import Terminal
from time import sleep

T = Terminal()

DEBUG = True

# color
# black
# red
# green
# yellow
# blue
# magenta
# cyan
# white


def log(*args, color="white", debug=DEBUG, delay=0):
    """
    log to the terminal

    kwargs:
        color: string; black, red, green, yellow, blue, magenta, cyan, white
        debug: boolean; to print or not. default set in log.py
        delay: number; seconds to pause after printing
    """
    if debug or color in ["red", "yellow"]:
        string = " ".join([getattr(T, color)(str(arg)) for arg in args])
        print(string)
    if delay:
        sleep(delay)

# debug.py - bion
# why?: organize helper functions for debugging

DEBUG = True


def log(*args, debug=DEBUG):
    if debug:
        print(*args)

# debug.py - bion
# why?: organize helper functions for debugging


def log(*args, debug=False):
    if debug:
        print(*args)

import random


def safe_sample(a, b):
    try:
        if a < b:
            return random.randint(a, b)
        elif b < a:
            return random.randint(b, a)
        else:
            return a
    except Exception as e:
        return a

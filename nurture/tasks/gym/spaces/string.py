import random
import string
import gym

class String(gym.Space):
    """ A space of strings """

    def __init__(
                self,
                length=None,
                min_length=1,
                max_length=180,
                min=32,
                max=127
            ):
        self.length = length
        self.min_length = min_length
        self.max_length = max_length
        self.letters = string.ascii_letters + " .,!-"

    def sample(self):
        length = random.randint(self.min_length, self.max_length)
        string = ""
        for i in range(length):
            letter = random.choice(self.letters)
            string += letter
        return string

    def contains(self, x):
        return type(x) is "str" and len(x) > self.min and len(x) < self.max

    def __repr__(self):
        return "String()"

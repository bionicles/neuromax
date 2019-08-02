import numpy as np
import string
import random
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


class Ragged(gym.Space):
    """ space with variable (None) dimensions for shape """

    def __init__(self, shape, variance=1., mean=0., high=None, low=None, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        # sampling
        self.variance = variance
        self.mean = mean
        # constraints
        self.high = high
        self.low = low

    def sample(self):
        if self.shape is not None and None in self.shape:
            raise ValueError("cannot sample arrays with unspecified dimensionality")
        else:
            if self.dtype in [np.float16, np.float32, np.float64]:
                sample = np.random.normal(self.mean, self.variance, self.shape)
            if self.dtype in [np.uint8, np.uint16, np.uint32, np.int8, np.int16, np.int32]:
                if self.low is None:
                    raise ValueError("cannot sample integers without at least low value")
                sample = np.random.randint(self.low, self.high, dtype=self.dtype)
        if self.low is not None and self.high is not None:
            sample = np.clip(sample, self.low, self.high)
        return sample

    def contains(self, x):
        if x.shape is self.shape and x.dtype is self.dtype:
            if self.low is not None and self.high is None:
                if all(self.low < x):
                    return True
            if self.low is None and self.high is not None:
                if all(x < self.high):
                    return True
            if self.low and self.high:
                if all(self.low < x < self.high):
                    return True
        return False

    def __repr__(self):
        return "Array(shape={})".format(str(self.shape))


class List(gym.Space):
    """ hold list of subspaces """

    def __init__(self, subspace, length=None):
        self.subspace = subspace
        self.length = length
        self.list = []

    def sample(self):
        if self.length is None:
            length = random.randint(0,3)
        else:
            length = self.length
        sample = []
        for i in range(length+1):
            subsample = self.subspace.sample()
            sample.append(subsample)
        self.list = sample
        return sample

    def contains(self, x):
        if x in self.list:
            return True
        else:
            return False

    def __repr__(self):
        if self.length is None:
            return "List({})".format(self.subspace.__repr__())
        else:
            return "List({}, length={})".format(self.subspace.__repr__(), self.length)

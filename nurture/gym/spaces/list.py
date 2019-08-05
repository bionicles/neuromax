import random
import gym


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

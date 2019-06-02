import numpy as np
import random
import gym

class Array(gym.Space):
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

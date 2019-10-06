# array.py - why?: allow arbitrary dimensionality spaces with None
import numpy as np
import gym


class Array(gym.Space):
    def __init__(
            self,
            shape,
            variance=1.,
            mean=0.,
            high=None,
            low=None,
            dtype=np.float32
            ):
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
            raise ValueError("cannot sample arrays with shape None")
        else:
            result = np.random.normal(self.mean, self.variance, self.shape)
            if self.high is not None or self.low is not None:
                result = np.clip(result, self.low, self.high)
            return result

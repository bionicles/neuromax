from model import ActorCriticModel
from queue import Queue
import tensorflow as tf
import threading
import gym
import numpy as np
import os

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

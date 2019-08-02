import tensorflow as tf
from PIL import Image
import numpy as np
import gym


class Coder:
    """stacked tied convolutional coder in numpy"""
    
    def __init__(self, tasks):
        self.envs = [gym.make(task) for task in tasks]
        self.observations = [env.reset() for env in self.envs]
        self.action_samples = [env.action_space.sample() for env in self.envs]

    def forward(self, sensation):
        print("Coder.forward")
        [print(key, "-", sensation[key] + "\n") for key in list(sensation.keys())]
        self.perceptions = [self.percieve(input) for input in sensation]
        self.reconstructions = [self.reconstruct(perception) for perception in self.perceptions]
        return self.perceptions, self.reconstructions

    def percieve(self, input):
        # strings might be images or
        if type(input) == "str":
            if input[-4:] in [".png", ".jpg"]:
                input_array = np.asarray(Image.open(input))
            else:
                input_array = np.array([[k, ord(character)] for k, character in enumerate(input)])
        else:
            input_array = input
        if len(input_array.shape) == 2:
            perception = tf.nn.conv2d(input_array, self.weights2, strides=1)
        else:
            perception = tf.nn.conv1d(input_array, self.weights1, strides=1)
        return perception

    def reconstruct(self, perception):
        if len(perception.shape) == 2:
            reconstruction = tf.nn.conv2d(perception, self.weights2.transpose, strides=1)
        else:
            reconstruction = tf.nn.conv1d(perception, self.weights1.transpose, strides=1)
        return reconstruction

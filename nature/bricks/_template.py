import tensorflow_probability as tfp
import tensorflow as tf

from nanoid import generate

import gin

tfd = tfp.distributions
tfpl = tfp.layers
K = tf.keras
L = K.layers

MY_PARAMETER_MIN, MY_PARAMETER_MAX = 1, 4
MY_CHOICES = ["cake", "pie"]


@gin.configurable  # <--- ez config status
class MyBrick:
    """
    MyBrick [DOES STUFF]
    because [REASON]

    Args:
    agent: Agent which holds this brick and has pull_choices/pull_numbers
    """

    def __init__(self, agent, brick_id):
        self.agent = agent
        self.brick_id = brick_id + "other_stuff" + generate()
        self.pull_numbers = agent.pull_numbers
        self.pull_choices = agent.pull_choices

    def build(self):
        self.pull_numbers("my_parameter", MY_PARAMETER_MIN, MY_PARAMETER_MAX)
        self.pull_choices("my_choice", MY_CHOICES)

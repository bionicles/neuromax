import tensorflow_probability as tfp
import tensorflow_addons as tfa
import tensorflow as tf
import gin

tfd = tfp.distributions
tfpl = tfp.layers
K = tf.keras
L = K.layers

MY_PARAMETER_MIN, MY_PARAMETER_MAX = 1, 4
MY_CHOICES = ["cake", "pie"]


@gin.configurable
class MyBrick:
    """
    MyBrick [DOES STUFF]
    because [REASON]

    Args:
    agent: Agent which holds this brick and has pull_choices/pull_numbers
    """

    def __init__(self, agent):
        self.agent = agent
        self.pull_numbers = agent.pull_numbers
        self.pull_choices = agent.pull_choices

    def build(self):
        self.pull_numbers("my_parameter", MY_PARAMETER_MIN, MY_PARAMETER_MAX)
        self.pull_choices("my_choice", MY_CHOICES)

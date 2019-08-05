MY_PARAMETER_MIN, MY_PARAMETER_MAX = 1, 4


class DenseCoder:
    """
    DenseCoder invertibly transforms inputs into some code and back
    to provide a generative latent model with fewer parameters

    Args:
    agent: Agent which holds this brick and has pull_choices/pull_numbers
    """

    def __init__(self, agent):
        self.agent = agent
        self.pull_numbers = agent.pull_numbers
        self.pull_choices = agent.pull_choices

    def build(self):
        self.pull_numbers("my_parameter", MY_PARAMETER_MIN, MY_PARAMETER_MAX)

# TransformerCoder ?
# Conv2DCoder

class DNC_RNN:
    """An RNN with DNC cells"""

    def __init__(self, agent, brick_id):
        self.agent = agent
        self.brick_id = brick_id
        self.pull_numbers = agent.pull_numbers
        self.pull_choices = agent.pull_choices
        # TODO: make a DNC RNN

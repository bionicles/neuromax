from tensorflow.keras.layers import Layer


class Brick(Layer):
    """A callable network module"""

    def __init__(self, agent, id, parts, call, out):
        for key in parts:
            setattr(self, key, parts[key])
        self.agent = agent
        self.parts = parts
        self.call = call
        self.out = call(out) if out else None
        self.id = id
        super(Brick, self).__init__()

    def build(self):
        pass

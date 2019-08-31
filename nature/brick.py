from tensorflow.keras.layers import Layer

from tools import package


class Brick(Layer):

    def __init__(self, *parts):
        parts = package(parts)
        assert "agent" in parts.keys()
        assert "call" in parts.keys()
        super(Brick, self).__init__()
        for key in parts:
            setattr(self, key, parts[key])
        self.agent.bricks[id] = self
        return self

    def build(self):
        self.built = True

    def __repr__(self):
        nl = '\n'
        kv = []
        kv.append(f" id:  {self.id}")
        kv.append("")
        for k, v in self.parts.items():
            if isinstance(v, Brick):
                string = f"  brick: {v.id}"
            elif k is not "id":
                string = f" {k}:  {v}"
            kv.append(string)
        kv.append("")
        substring = nl.join([str(x) for x in kv])
        return f"""

-------------------------------- Brick --------------------------------
{substring}
-----------------------------------------------------------------------

"""

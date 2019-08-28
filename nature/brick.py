from tensorflow.keras.layers import Layer

import nature


class Brick(Layer):
    """A callable network module"""

    def __init__(self, agent, parts):
        self.agent = agent
        parts = getattr(nature, f"use_{parts['brick_type']}")(agent, parts)
        for key in parts:
            setattr(self, key, parts[key])
        super(Brick, self).__init__()
        self.agent.graph[id] = self

    def build(self):
        pass

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

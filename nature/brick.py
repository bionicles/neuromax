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

    def __repr__(self):
        nl = '\n'
        kv = []
        kv.append(f" id:  {self.id}")
        kv.append("")
        for k, v in self.parts.items():
            if isinstance(v, Brick):
                string = f"  brick: {v.id}"
            else:
                string = f" {k}:  {v}"
            kv.append(string)
        kv.append("")
        kv.append(f" out:  {self.out}")
        substring = nl.join([str(x) for x in kv])
        return f"""

-------------------------------- Brick --------------------------------
{substring}
-----------------------------------------------------------------------

"""

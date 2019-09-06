STYLE = "filled"


def add_node(G, id, color, shape, node_type, output=None, spec=None):
    label = None if color is 'black' else id
    G.add_node(
        id, label=label,
        style=STYLE, color=color, shape=shape,
        node_type=node_type, output=output, spec=spec)

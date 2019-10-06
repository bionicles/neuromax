COLOR = "black"
STYLE = "filled"
SHAPE = "square"

def add_node(G, id, color, shape, node_type, output=None, spec=None, n=None):
    label = None if color is 'black' else id
    G.add_node(
        id, label=label,
        style=STYLE, color=color, shape=shape,
        node_type=node_type, output=output, spec=spec, n=n)

def insert_node(G, id1=None, id2=None):
    if id1 is None and id2 is None:
        id1 = random.choice(list(G.nodes()))
        successors = list(G.successors(id1))
        if len(successors) < 1:
            return
        id2 = random.choice(successors)
    log("split_edge", id1, id2)
    if is_blacklisted(id1) and is_blacklisted(id2) or id1 == id2:
        return
    new_id = f"{id1}--->{id2}"
    log(f"{id1}--->{new_id}--->{id2}")
    add_node(G, new_id, COLOR, SHAPE, "brick")
    G.add_edge(id1, new_id)
    G.add_edge(new_id, id2)


def insert_nodes(G):
    cutoff = G.order() * 1.1
    while G.order() < cutoff:
        insert_node(G)

def delete_edge():
    global G
    edge = random.choice(list(G.edges()))
    log("delete_edge", edge)
    for node in edge:
        if is_blacklisted(node):
            log("not deleting", edge, "because", node, "is blacklisted")
            return
    G.remove_edge(*edge)
    return edge


def delete_node():
    global G
    node_list = list(G.nodes())
    n_boxes = count_boxes()
    if n_boxes == 1:
        return
    id = random.choice(node_list)
    if is_blacklisted(id):
        return
    successors = G.successors(id)
    for successor in successors:
        try:
            if "_out" in successor:
                G.remove_node(f"{id}_out")
                G.remove_node(f"{id}_feedback")
                break
        except Exception as e:
            log(e)
    predecessors = G.predecessors(id)
    successors = G.successors(id)
    for predecessor in predecessors:
        for successor in successors:
            if predecessor == successor:
                continue
            G.add_edge(predecessor, successor)
    G.remove_node(id)
    return id


def decimate(p=0.1):
    """ reduce the number of nodes and edges """
    global G
    n_boxes = count_boxes()
    if n_boxes < (1 / p) or G.size() < (1 / p):
        return
    n_boxes *= (1 - p)
    victims = []
    loops = 0
    while count_boxes() > n_boxes:
        log("DELETE A NODE")
        victim = delete_node()
        victims.append(victim)
        loops += 1
        if loops > 100:
            return
    n_edges = G.size() * (1 - p)
    loops = 0
    while G.size() > n_edges:
        log("DELETE AN EDGE")
        victim = delete_edge()
        victims.append(victim)
        loops += 1
        if loops > 100:
            return
    [log("KILLED", victim) for victim in victims]

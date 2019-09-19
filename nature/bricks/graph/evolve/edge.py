import networkx as nx

from nature import is_blacklisted

def insert_edge(G, node1=None, node2=None):
    prior_cycles = list(nx.simple_cycles(G))
    sample = safe_sample(G.nodes(), 2)
    log(f"SAMPLE, {sample}")
    for node in sample:
        if is_blacklisted(node):
            return
    node1, node2 = sample
    if node2 in G.successors(node1):
        return
    G.add_edge(node1, node2)
    new_cycles = list(nx.simple_cycles(G))
    if len(new_cycles) == len(prior_cycles) and node1 != node2 and node2 != node1:
        return G
    else:
        handle_feedback(node1, node2)

def insert_edges(G, p=0.2):
    if count_boxes() < 10:
        return
    n_edges = G.size() * (1 + p)
    loops = 0
    while G.size() < n_edges:
        log("CONNECT!")
        add_edge()
        loops += 1
        if loops > 100:
            return

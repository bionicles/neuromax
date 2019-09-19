import networkx as nx
import random

from tools import safe_sample, log
from nature import Regulon

MIN_P_INSERT, MAX_P_INSERT = 0.1618, 0.42


def insert_motif(G, id):
    M, new_ids = Regulon(parent=id)
    G = nx.compose(G, M)
    predecessors, successors = G.predecessors(id), new_ids[0]
    for predecessor in predecessors:
        for successor in successors:
            G.add_edge(predecessor, successor)
    predecessors, successors = new_ids[-1], list(G.successors(id))
    for predecessor in predecessors:
        for successor in successors:
            G.add_edge(predecessor, successor)
    G.remove_node(id)
    return G


def insert_motifs(G):
    p_insert = safe_sample(MIN_P_INSERT, MAX_P_INSERT)
    nodes = G.nodes(data=True)
    for node in nodes:
        try:
            if node[1]["shape"] is "square" and random.random() < p_insert:
                insert_motif(G, id=node[0])
        except Exception as e:
            log('insert_motifs Exception', e)
    return G

import networkx as nx

from tools import safe_sample
from nature import add_node

MIN_LAYERS, MAX_LAYERS = 1, 3
MIN_NODES, MAX_NODES = 1, 2


def Regulon(parent=None):
    n_layers = safe_sample(MIN_LAYERS, MAX_LAYERS)
    M, ids = nx.MultiDiGraph(), []
    for layer_number in range(n_layers):
        n_nodes = safe_sample(MIN_NODES, MAX_NODES)
        ids_of_nodes_in_this_layer = []
        for node_number in range(n_nodes):
            node_id = f"{layer_number}.{node_number}"
            if parent:
                node_id = f"{parent}.{node_id}"
            add_node(M, node_id, "black", "square", "brick")
            ids_of_nodes_in_this_layer.append(node_id)
        ids.append(ids_of_nodes_in_this_layer)
    for predecessor_layer_number, predecessor_node_ids in enumerate(ids):
        for successor_layer_number, successor_node_ids in enumerate(ids):
            if predecessor_layer_number < successor_layer_number:
                for predecessor_node_id in predecessor_node_ids:
                    for successor_node_id in successor_node_ids:
                        M.add_edge(predecessor_node_id, successor_node_id)
    return M, ids

def is_blacklisted(G, id):
    return G.node[id]['shape'] != "square"

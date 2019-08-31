import networkx as nx
import os

from tools import log


def screenshot_graph(G, folderpath, filename):
    """Make a png image of a graph."""
    imagepath = os.path.join(folderpath, f"{filename}.png")
    log(f"SCREENSHOT {imagepath} with {G.order()} nodes, {G.size()} edges")
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update()
    A.draw(path=imagepath, prog="dot")

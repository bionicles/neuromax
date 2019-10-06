import networkx as nx
import os

from tools import log

FOLDER = './pngs'


def screenshot_graph(G, filename, folder=FOLDER):
    """Make a png image of a graph."""
    imagepath = os.path.join(folder, f"{filename}.png")
    log(f"SCREENSHOT {imagepath} with {G.order()} nodes, {G.size()} edges")
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update()
    A.draw(path=imagepath, prog="dot")

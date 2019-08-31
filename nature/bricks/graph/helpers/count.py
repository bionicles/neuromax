def count_boxes(G):
    return len(
        [n for n in list(G.nodes(data=True))
         if n[1]["shape"] is "square"]
        )

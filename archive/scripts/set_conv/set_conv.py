from itertools import combinations
from datetime import datetime
from natsort import natsorted
import networkx as nx
import imageio
import os

MAX_INPUTS, MAX_OUTPUTS = 6,6
PIC_PATH = "archive/scripts/set_conv/pics"
log = print

graph_number = 1


def conv_set(inputs=4, input_set_size=4, outputs=6, output_set_size=2):
    global graph_number
    G = nx.MultiDiGraph()
    for n in range(inputs):
        G.add_node(f"input-{n}", label="", color="blue", style="filled", shape="circle")
    for m in range(outputs):
        G.add_node(f"output-{m}", label="", color="red", style="filled", shape="triangle")
    N_set_keys = []
    for set in combinations(range(inputs), input_set_size):
        set_node_key = f"input-set-" + '-'.join([str(s) for s in set])
        N_set_keys.append(set_node_key)
        G.add_node(set_node_key, label="", shape="square", style="filled", color="black")
        [G.add_edge(f"input-{n}", set_node_key) for n in set]
    M_set_keys = []
    for set in combinations(range(outputs), output_set_size):
        set_node_key = f"output-set-" + '-'.join([str(s) for s in set])
        M_set_keys.append(set_node_key)
        G.add_node(set_node_key, label="", shape="square", style="filled", color="black")
        [G.add_edge(set_node_key, f"output-{m}") for m in set]
    for N_set_key in N_set_keys:
        for M_set_key in M_set_keys:
            G.add_edge(N_set_key, M_set_key)
    G.name = f"{graph_number}_{inputs}_choose_{input_set_size}_to_{outputs}_choose_{output_set_size}"
    # G.name = f"{inputs}-{outputs}-{G.size()}-{graph_number}"
    # G.name = f"{G.size()}-{graph_number}"
    graph_number += 1
    return G

# for m in M:
# get_outputs(m)
# for

def make_gif(folder_path="archive/scripts/set_conv/pics", out_path="set_conv/gifs",
             image_x=1024, image_y=512, brand_y=64,
             brand="Bit Pharma", movie=True, framerate=12):
    image_size = f"{image_x}x{image_y}"
    dirfiles = []
    for filename in os.listdir(folder_path):
        log("parsed", filename)
        name, ext = os.path.splitext(filename)
        if name == 'model' or ext != ".png":
            continue
        log("convert", filename, "to jpg")
        cmd_string = f"convert '{folder_path}/{name}.png' -gravity Center -thumbnail '{image_size}' -extent {image_size} -font Courier -gravity Center -undercolor '#00000080' -size {image_x}x{brand_y} -fill white caption:'{name}' +swap -append -gravity Center '{folder_path}/{name}.jpg'"
        os.system(cmd_string)
        dirfiles.append(f"{folder_path}/{name}.jpg")
    dirfiles = natsorted(dirfiles)
    [log(dirfile) for dirfile in dirfiles]
    now = str(datetime.now()).replace(" ", "_")
    if movie:
        movie_path = f"archive/movies/net-{now}.mp4"
        writer = imageio.get_writer(movie_path, fps=framerate)
        [writer.append_data(imageio.imread(dirfile)) for dirfile in dirfiles]
        writer.close()
        log(f"SAVED MOVIE TO {movie_path}")
    else:
        images = [imageio.imread(dirfile) for dirfile in dirfiles]
        gif_path = os.path.join(out_path, f"net-{now}.gif")
        imageio.mimsave(gif_path, images, duration=(1/framerate))
        log(f"SAVED GIF TO {gif_path}")


def screenshot_graph(G, folderpath, filename):
    """Make a png image of a graph."""
    imagepath = os.path.join(folderpath, f"{filename}.png")
    print(f"SCREENSHOT {imagepath} with {G.order()} nodes, {G.size()} edges")
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update()
    A.draw(path=imagepath, prog="dot")


def empty_folder():
    filelist = [f for f in os.listdir(PIC_PATH)]
    for f in filelist:
        os.remove(os.path.join(PIC_PATH, f))


def main():
    empty_folder()
    for N in range(1, MAX_INPUTS + 1):
        for i in range(1, N+1):
            for M in range(1, MAX_OUTPUTS + 1):
                for j in range(1, M+1):
                    print(N, i, M, j)
                    G = conv_set(inputs=N, input_set_size=i, outputs=M, output_set_size=j)
                    screenshot_graph(G, PIC_PATH, G.name)
    make_gif()


main()

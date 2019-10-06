import networkx
import imageio
import random
import shutil
import time
import gym

tasks = {
    "BipedalWalker-v2": {
        "intelligence": "bodily-kinesthetic",
        "goal": "learn to walk"
    },
    "MontezumaRevenge-v0": {
        "intelligence": "curiousity-exploration",
        "goal": "avoid traps and find items"
    },
}


def get_items(tasks):
    task_names = list(tasks.keys())
    for task_name in task_names:
        print("---------")
        print("get items for", task_name)
        print("---------")
        try:
            task = tasks[task_name]
            env = gym.make(task_name)
            observation = env.reset()
            observation = wrap_if_not_dict(observation, task_name)
            observation["goal"] = task_name + ": " + task["intelligence"] + " " + task["goal"]
            observation_items = [
                [key, observation[key], task_name]
                for key in list(observation.keys())
                ]
            if type(env.action_space).__name__ is "Dict":
                action_items = [
                    [space[0], space[1], task_name]
                    for space in env.action_space.spaces.items()
                ]
            else:
                action_items = [[task_name+"-action", env.action_space, task_name]]
            tasks[task_name]["observation_items"] = observation_items
            tasks[task_name]["action_items"] = action_items
        except:
            tasks[task_name]["excluded"] = True
            print('failed to fetch', task_name)
    return tasks


def prepare(tasks):
    task_names = list(tasks.keys())
    for task_name in task_names:
        print("---------")
        print("prepare", task_name)
        print("---------")
        try:
            task_dict = tasks[task_name]
            items = []
            env = gym.make(task_name)
            observation = env.reset()
            observation = wrap_if_not_dict(observation, task_name)
            # observation/action, title str array, shape, dtype, low, high
            for key in list(observation.keys()):
                space = observation[key]
                item = space2item("observation", task_name, key, space)
                items.append[item]
            if type(env.action_space).__name__ is "Dict":
                for key in list(env.action_space.spaces.items()):
                    item = space2item("action", task_name, space)
                    items.append(item)
                else:
                    space = env.action_space
                    item = space2item("action", task_name, space)
                    items.append(item)
            tasks[task_name]["excluded"] = False
        except:
            print("failed to fetch", task_name)
            tasks[task_name]["excluded"] = True
    return tasks


def space2item(action_or_observation, task_name, item_name, space):
    # get the type
    space_type = type(space)
    type_name = space_type.__name__
    if type_name == "str":
        value = str2vec(space)
        shape = value.shape
        dtype = value.dtype
        high = 126
        low = 32
    if type_name == "Discrete":
        dtype = "np.int64"
        value = space
        shape = (1)
        high = space.n - 1
        low = 0
    if type_name == "int":
        value = space
        shape = (1)
        high = None
        low = None
    if type_name == "Array" or type_name == "ndarray" or type_name == "Box":
        value = space
        shape = space.shape
        dtype = type_name
    if type_name == "List":
        value = np.array(space)
        shape = value.shape
        high = None
        low = None
    item = {
        "action_or_observation": action_or_observation,
        "key": str2vec(task_name + " " + item_name),
        "value": value,
        "shape": value.shape,
        "dtype": value.dtype,
        "high": high,
        "low": low
        }
    return item


def get_separate_boxes(tasks):
    G = networkx.MultiDiGraph()
    G.add_node("source", label="", shape="circle", style="filled", color="red")
    G.add_node("sink", label="", shape="triangle", style="filled", color="blue")
    # add shared inputs
    G.add_node("shared-in", label="", shape="circle", style="filled", color="yellow")
    G.add_edge("source", "shared-in")
    G.add_node("reward", label="", shape="star", style="filled", color="yellow")
    G.add_edge("shared-in", "reward")
    G.add_node("t-1", label="", shape="cylinder", style="filled", color="yellow")
    G.add_edge("shared-in", "t-1")
    # add shared outputs
    G.add_node("shared-out", label="", shape="circle", style="filled", color="yellow")
    G.add_edge("shared-out", "sink")
    G.add_node("reward-prediction", label="", shape="star", style="filled", color="yellow")
    G.add_edge("reward-prediction", "shared-out")
    G.add_node("t", label="", shape="cylinder", style="filled", color="yellow")
    G.add_edge("t", "shared-out")
    task_names = list(tasks.keys())
    for task_name in task_names:
        try:
            if tasks[task_name]["excluded"] is True:
                continue
        except:
            print(task_name, "is not excluded -- adding to the graph")
        task_input_node_name = task_name + "-in"
        print("add node", task_input_node_name)
        G.add_node(task_input_node_name, label="", shape="circle", style="filled", color="red")
        G.add_edge("source", task_input_node_name)
        task_box_name = task_name+"-box"
        print("add node", task_box_name)
        G.add_node(task_box_name, label="", style="filled", shape="square", color="black")
        task_output_node_name = task_name + "-out"
        print("add node", task_output_node_name)
        G.add_node(task_output_node_name, label="", shape="triangle", style="filled", color="blue")
        G.add_edge(task_output_node_name, "sink")
#
        split_edge(G, "reward", task_box_name, color="black", shape="diamond", style="filled")
        split_edge(G, "t-1", task_box_name, color="black", shape="diamond", style="filled")
        split_edge(G, task_box_name, "reward-prediction", color="black", shape="diamond", style="filled")
        split_edge(G, task_box_name, "t", color="black", shape="diamond", style="filled")
#
        observation_items = tasks[task_name]["observation_items"]
        for item in observation_items:
            print("adding", item)
            # add the data to the input
            node_name = task_name+"-"+item[0]+"-observation"
            G.add_node(node_name, label="", shape="circle", style="filled", color="red")
            G.add_edge(task_input_node_name, node_name)
            split_edge(G, node_name, task_box_name, color="black", shape="diamond", style="filled")
            # add the reconstruction to the output
            node_name = task_name+"-"+item[0]+"-reconstruction"
            G.add_node(node_name, label="", shape="triangle", style="filled", color="blue")
            split_edge(G, task_box_name, node_name, color="black", shape="diamond", style="filled")
            G.add_edge(node_name, task_output_node_name)
#
        action_items = tasks[task_name]["action_items"]
        for item in action_items:
            print("adding", item)
            print("item", item)
            # add the action to the input
            node_name = task_name+"-"+item[0]+"-action-space-shape"
            print("add node", node_name)
            G.add_node(node_name, label="", shape="circle", style="filled", color="red")
            G.add_edge(task_input_node_name, node_name)
            split_edge(G, node_name, task_box_name, color="black", shape="diamond", style="filled")
            # add the action to the output
            node_name = task_name+"-"+item[0]
            print("add node", node_name)
            G.add_node(node_name, label="", shape="triangle", style="filled", color="blue")
            split_edge(G, task_box_name, node_name, color="black", shape="diamond", style="filled")
            G.add_edge(node_name, task_output_node_name)
            print("add node", "reward-prediction")
    return G


def split_edge(G, source, sink, color="blue", style="filled", shape="diamond"):
    route_node_name = "{}->{}".format(source, sink)
    print("source", source, "route", route_node_name, "sink", sink)
    G.add_node(route_node_name, label="",  style=style, color=color, shape=shape, size=0.2)
    G.add_edge(source, route_node_name)
    G.add_edge(route_node_name, sink)
    return route_node_name


def wrap_if_not_dict(item, task_name):
    if item.__class__.__name__ not in ["dict", "OrderedDict"]:
        print("wrapping", task_name, item.__class__.__name__)
        return {task_name: item}
    return item


def get_regulon(parent=None, layers=None):
    num_layers = random.randint(MIN_LAYERS, MAX_LAYERS)
    G = networkx.MultiDiGraph()
    names = []
    for layer_number in range(num_layers):
        num_nodes = random.randint(MIN_NODES, MAX_NODES)
        print("layer number", layer_number, "has", num_nodes, "nodes")
        names_of_nodes_in_this_layer = []
        for node_number in range(num_nodes):
            if parent is not None:
                node_name = "{}.{}.{}".format(parent, layer_number, node_number)
            else:
                node_name = "{}.{}".format(layer_number, node_number)
            print("add node", node_name)
            G.add_node(node_name, label="", style="filled", shape="square", color="black")
            names_of_nodes_in_this_layer.append(node_name)
        names.append(names_of_nodes_in_this_layer)
    for predecessor_layer_number, predecessor_node_names in enumerate(names):
        for successor_layer_number, successor_node_names in enumerate(names):
            if predecessor_layer_number >= successor_layer_number:
                continue
            print("predecessor", predecessor_layer_number, predecessor_node_names)
            print("successor", successor_layer_number, successor_node_names)
            for predecessor_node_name in predecessor_node_names:
                for successor_node_name in successor_node_names:
                    print("adding edge from", predecessor_node_name, "to", successor_node_name)
                    G.add_edge(predecessor_node_name, successor_node_name)
    return G, names


def insert_motif(G, name, motif):
    # get a motif
    Gi, new_names = get_regulon(name)
    # add the motif to the graph
    Gn = networkx.compose(G, Gi)
    # point the prececessors of the node to replace at all nodes in first layer
    predecessors = Gn.predecessors(name)
    successors = new_names[0]
    for predecessor in predecessors:
        for successor in successors:
            print("adding edge from", predecessor, "to", successor)
            Gn.add_edge(predecessor, successor)
    # point the last node in the motif at the successors of the node to replace
    predecessors = new_names[len(new_names)-1]
    successors = list(G.successors(name))
    for predecessor in predecessors:
        for successor in successors:
            print("adding edge from", predecessor, "to", successor)
            Gn.add_edge(predecessor, successor)
    # remove the node
    Gn.remove_node(name)
    return Gn


def screenshot(G, step):
    print("SCREENSHOT", step)
    A = networkx.nx_agraph.to_agraph(G)
    A.graph_attr.update(rankdir="LR")
    A.draw(path="./nets/{}.png".format(step), prog="dot")


def folder2gif():
    cmd_string = "convert '{}/{}' -gravity North -thumbnail '{}' -extent {} -font Courier -gravity North -undercolor '#00000080' -size 1024x32 -fill white caption:'Bit Pharma' +swap -append -gravity North '{}.jpg'".format("nets", "initial.png", IMAGE_SIZE, IMAGE_SIZE, "i")  -gravity North
    os.system(cmd_string)
    dirfiles = []
    for filename in os.listdir("nets"):
        if filename is "initial.png":
            continue
        name, ext = os.path.splitext(filename)
        print("convert", filename, "to jpg")
        cmd_string = "convert '{}/{}.png' -gravity North -thumbnail '{}' -extent {} -font Courier -gravity North -undercolor '#00000080' -size 1024x32 -fill white caption:'Bit Pharma' +swap -append -gravity North 'net-jpgs/{}.jpg'".format("nets", name, IMAGE_SIZE, IMAGE_SIZE, name)  -gravity center
        os.system(cmd_string)
        dirfiles.append("net-jpgs/{}.jpg")
    dirfiles = os.listdir("net-jpgs")
    dirfiles = [dirfile for dirfile in dirfiles if dirfile != "initial.jpg"]
    dirfiles = natsorted(dirfiles)
    [print(dirfile) for dirfile in dirfiles]
    initial = imageio.imread("i.jpg")
    images = []
    images.append(initial)
    for dirfile in dirfiles:
        versions = dirfile.split(".")
        if dirfile[-5] == str(1) and versions[1] != "2":
            images.append(initial)
        image = imageio.imread("net-jpgs/"+dirfile)
        images.append(image)
    gif_name = "net-{}.gif".format(str(time.time()))
    imageio.mimsave(gif_name, images)


def recurse(attempt, G, augmented_tasks):
    for step in range(STEPS):
        nodes = G.nodes(data=True)
        for node in nodes:
            try:
                if node[1]["shape"] is "square":
                    if random.random() < 0.64:
                        motif = "regulon"
                        G = insert_motif(G, node[0], motif)
            except Exception as e:
                print('exception in recurse', e)
                print("no shape for node", node)
        image_name = "{}.{}.{}".format(attempt, 1, step+1)
        screenshot(G, image_name)
    # depth first topological sort:
    if SORT is "depth":
        topo = networkx.topological_sort(G)
        step = STEPS
        for node in topo:
            if G.nodes[node]["color"] is "black":
                print("coloring", node)
                G.nodes[node]["color"] = "green"
                image_name = "{}.{}.{}".format(str(k+1), 2, str(step+1))
                screenshot(G, image_name)
                step += 1
    # breadth first topological sort:
    if SORT is 'breadth':
        frontier = ["source"]
        task_names = list(augmented_tasks.keys())
        for task_name in task_names:
            task_input_node_name = task_name + "-in"
            [frontier.append(edge[1]) for edge in G.out_edges(task_input_node_name)]
        next = []
        step = STEPS
        while len(frontier) > 0:
            for parent in frontier:
                children = [edge[1] for edge in G.out_edges(parent)]
                for child in children:
                    if child not in next and all([G.nodes[edge[0]]["color"] != "black" for edge in G.in_edges(child)]):
                        G.nodes[child]["color"] == "black"
                        next.append(child)
            for node_name in next:
                node = G.nodes[node_name]
                if node["color"] == "black":
                    print("coloring", node_name)
                    node["color"] = "green"
            frontier = next
            next = []
            image_name = "{}.{}.{}".format(attempt, 2, step+1)
            screenshot(G, image_name)
            step += 1
    return G


ATTEMPTS, STEPS, MIN_LAYERS, MAX_LAYERS, MIN_NODES, MAX_NODES = 1, 2, 1, 3, 1, 2
SORT = "breadth"  "breadth", "depth", "none"
IMAGE_SIZE = "{}x{}".format("1024", "512")
augmented_tasks = get_items(tasks)
observation_types = {}
action_types = {}
shapes = []


shutil.rmtree("nets")
shutil.rmtree("net-jpgs")
os.mkdir("nets")
os.mkdir("net-jpgs")
for k in range(ATTEMPTS):
    G_init = get_separate_boxes(augmented_tasks)
    screenshot(G_init, "initial")
    G = recurse(k+1, G_init, augmented_tasks)
folder2gif()

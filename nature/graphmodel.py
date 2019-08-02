# graphmodel.py - bion and kamel, july 2019
# why?: experiment faster with recursive neural architecture search

# more options for layers:
# Echo State Network: https://github.com/cameronosmith/Echo-State-Network/blob/master/EchoStateNetwork.py
# Graph: https://github.com/CyberZHG/keras-gcn/blob/master/keras_gcn/layers.py / https://github.com/tkipf/keras-gcn/blob/master/kegra/layers/graph.py
# Polymorphic: https://github.com/dip-scd/PolymorphicDense/blob/master/src/polymorphic_dense.py
# MOE: https://github.com/eminorhan/mixture-of-experts/blob/master/DenseMoE.py
# NALU: https://arxiv.org/pdf/1808.00508.pdf
# MDN: https://github.com/cpmpercussion/keras-mdn-layer/blob/master/mdn/__init__.py
import tensorflow_addons as tfa
from attrdict import AttrDict
from datetime import datetime
from natsort import natsorted
import tensorflow as tf
import networkx as nx
import numpy as np
import imageio
import random
import gym
import os

from .bricks import Transformer, KConvSet, get_kernel
B, L, K = tf.keras.backend, tf.keras.layers, tf.keras
InstanceNormalization = tfa.layers.InstanceNormalization

IMAGE_PATH = "./archive/nets"
IMAGE_X, IMAGE_Y = 1024, 1024
IMAGE_SIZE = f"{IMAGE_X}x{IMAGE_Y}"
BRAND_Y = 64
TRIALS_PER_MOVIE = 4
MODELS_PER_TRIAL = 1
MAKE_MOVIE = True
DURATION = 1.  # / 60.
ALWAYS_DECIMATE_IF_MORE_THAN = 64  # boxes
CLEAN_UP = True
DEBUG = True


def log(*args):
    if DEBUG:
        print(*args)


def safe_sample(a, b):
    try:
        if a < b:
            return random.randint(a, b)
        elif b < a:
            return random.randint(b, a)
        else:
            return a
    except Exception as e:
        return a


def is_blacklisted(id):
    log(f"is {id} blacklisted?")
    try:
        node = G.node[id]
        if node["shape"] != "square":
            log("yup!")
            return True
        else:
            log("nope!")
            return False
    except Exception as e:
        log("nope!")
        return False


def count_boxes():
    return len([node for node in list(G.nodes(data=True)) if node[1]["shape"] == "square"])


def get_initial_graph(prepared, initial_boxes):
    """Create a graph connecting task inputs to outputs with a black box."""
    G = nx.MultiDiGraph()

    G.add_node("source", label="SOURCE", shape="cylinder", color="gold")
    G.add_node("sink", label="SINK", shape="cylinder", color="gold")

    for task in prepared:
        task_source_key = f"{task.name}-source"
        G.add_node(task_source_key, label=task_source_key, shape="cylinder", color="gold")
        G.add_edge("source", task_source_key)
        task_sink_key = f"{task.name}-sink"
        G.add_node(task_sink_key, label=task_sink_key, shape="cylinder", color="gold")
        G.add_edge(task_sink_key, "sink")
        for i, input in enumerate(task.inputs):
            input_key = f"{task.name}-input-{i}"
            G.add_node(input_key, label=input_key, shape="circle", color="blue",
                       node_type="input", input=input)
            G.add_edge(task_source_key, input_key)
            encoder_key = f"{input_key}-encoder"
            G.add_node(encoder_key, label=encoder_key, shape="diamond", color="black",
                       node_type="encoder", input=input, output=CODE)
            G.add_edge(input_key, encoder_key)
            for box_number in range(initial_boxes):
                if box_number not in G.nodes():
                    G.add_node(box_number, label=box_number, shape="square", color="black")
                G.add_edge(encoder_key, box_number)
        for o, output in enumerate(task.outputs):
            output_key = f"{task.name}-output-{o}"
            G.add_node(output_key, label=output_key, shape="triangle", color="red",
                       node_type="output", output=output)
            G.add_edge(output_key, task_sink_key)
            decoder_key = f"{output_key}-decoder"
            G.add_node(decoder_key, label=decoder_key, shape="diamond", color="black",
                       node_type="decoder", input=CODE, output=output)
            for box_number in range(initial_boxes):
                G.add_edge(box_number, decoder_key)
            G.add_edge(decoder_key, output_key)
    return G


def split_edge(id1=None, id2=None, style="filled", color="black", shape="square"):
    global G
    if id1 is None and id2 is None:
        id1 = random.choice(list(G.nodes()))
        successors = list(G.successors(id1))
        if len(successors) < 1:
            return
        id2 = random.choice(successors)
    log("split_edge", id1, id2)
    if is_blacklisted(id1) and is_blacklisted(id2) or id1 == id2:
        return
    new_node_id = f"{id1}--->{id2}"
    log(f"{id1}--->{new_node_id}--->{id2}")
    G.add_node(new_node_id, label="", style=style, color=color, shape=shape)
    G.add_edge(id1, new_node_id)
    G.add_edge(new_node_id, id2)


def split_edges():
    cutoff = G.order() * 1.1
    while G.order() < cutoff:
        split_edge()


def get_regulon(hp, parent=None, layers=None):
    """Build a subgraph to replace a node."""
    num_layers = safe_sample(hp.min_layers, hp.max_layers)
    M = nx.MultiDiGraph()
    ids = []
    for layer_number in range(num_layers):
        n_nodes = safe_sample(hp.min_nodes, hp.max_nodes)
        log(f"add layer {layer_number} with {n_nodes} nodes")
        ids_of_nodes_in_this_layer = []
        for node_number in range(n_nodes):
            if parent:
                node_id = f"{parent}.{layer_number}.{node_number}"
            else:
                node_id = f"{layer_number}.{node_number}"
            log(f"add node {node_id}")
            M.add_node(node_id, label="", style="filled", shape="square", color="black")
            ids_of_nodes_in_this_layer.append(node_id)
        ids.append(ids_of_nodes_in_this_layer)
    for predecessor_layer_number, predecessor_node_ids in enumerate(ids):
        for successor_layer_number, successor_node_ids in enumerate(ids):
            if predecessor_layer_number >= successor_layer_number:
                continue
            for predecessor_node_id in predecessor_node_ids:
                for successor_node_id in successor_node_ids:
                    log(f"{predecessor_node_id}--->{successor_node_id}")
                    M.add_edge(predecessor_node_id, successor_node_id)
    return M, ids


def insert_motif(hp, id=None):
    """Replace a node with a subgraph."""
    global G
    id = random.choice(list(G.nodes())) if id is None else id
    if is_blacklisted(id):
        return
    M, new_ids = get_regulon(hp, parent=id)
    G = nx.compose(G, M)
    # connect the node's predecessors to all nodes in first layer
    predecessors = G.predecessors(id)
    successors = new_ids[0]
    for predecessor in predecessors:
        for successor in successors:
            log(f"{predecessor}--->{successor}")
            G.add_edge(predecessor, successor)
    # connect the last layer of the motif to the node's successors
    predecessors = new_ids[-1]
    successors = list(G.successors(id))
    for predecessor in predecessors:
        for successor in successors:
            log(f"{predecessor}--->{successor}")
            G.add_edge(predecessor, successor)
    G.remove_node(id)


def recurse(hp):
    """with probability P, replace nodes with random feedforward regulons"""
    global G
    nodes = G.nodes(data=True)
    p_insert = hp.p_insert if count_boxes() > hp.initial_boxes else 1.
    for node in nodes:
        try:
            if node[1]["shape"] is "square":
                if random.random() < p_insert:
                    insert_motif(hp, id=node[0])
        except Exception as e:
            log('exception in recurse', e)


def handle_feedback(id1, id2):
    input_key = f"{id1}_feedback"
    if input_key not in G.successors("source"):
        G.add_node(input_key, label=input_key, color="blue", shape="circle", node_type="input")
        G.add_edge("source", input_key)
    if input_key not in G.predecessors(id2):
        G.add_edge(input_key, id2)
    output_key = f"{id1}_out"
    if output_key not in G.successors(id1):
        G.add_node(output_key, label=output_key, color="red", shape="triangle", node_type="recurrent")
        G.add_path([id1, output_key, "sink"])
    G.remove_edge(id1, id2)


def add_edge(node1=None, node2=None):
    global G
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


def connect(p=0.2):
    global G
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


def do_nothing():
    log("did nothing")


def clean_up():
    global G
    for id in list(G.nodes()):
        if G.node[id]["shape"] == "square":
            predecessors = G.predecessors(id)
            if len([*predecessors]) < 1:
                log("reconnecting", id, "to input")
                G.add_edge("input", id)
            successors = G.successors(id)
            if len([*successors]) < 1:
                log("reconnecting", id, "to output")
                G.add_edge(id, "output")
    inputs = list(G.successors("source"))
    for input in inputs:
        successors = list(G.successors(input))
        if len(successors) < 1:
            G.remove_node(input)
            G.remove_node(f"{input[:-9]}_out")
            continue
        for successor in successors:
            if G.node[successor]["shape"] == "triangle":
                log("CLEANUP BYPASS", input, successor)
                G.remove_edge(input, successor)
        predecessors = list(G.predecessors(input))
        for predecessor in predecessors:
            if G.node[predecessor]["shape"] == "triangle":
                log("CLEANUP BYPASS", predecessor, input)
                G.remove_edge(predecessor, input)


def mutate(hp):
    boxes = count_boxes()
    log(f"{boxes} boxes")
    if hp.force_recursion == 1 and k is 0:
        log("forced recursion!")
        recurse(hp)
    # elif boxes > ALWAYS_DECIMATE_IF_MORE_THAN:
    #     log("forced decimation!")
    #     decimate(0.2)
    else:
        mutation = np.random.choice([recurse, decimate, connect, split_edges, insert_motif,
                          add_edge, delete_edge, delete_node, split_edge, do_nothing],
                         1, p=hp.mutation_distribution).item(0)
        if mutation in [recurse, insert_motif]:
            mutation(hp)
        else:
            mutation()
    if CLEAN_UP:
        clean_up()

def differentiate(hp):
    """Assign ops and arguments to nodes."""
    if hp.force_skip:
        G.add_edge("input", "output")
    for node in G.nodes(data=True):
        node_id, node_data = node
        log("differentiate", node_id, node_data)
        node_data["output"] = None
        node_data["op"] = None
        if node_data["shape"] is "square" or "output" in node_id:
            if node_id == "output":
                d_out = node_data["output_shape"][-1]
                node_type = hp.last_layer
                activation = "tanh"
            else:
                node_type = str(np.random.choice(['sepconv1d', 'transformer',
                                                  'k_conv1', 'k_conv2', 'k_conv3',
                                                  "deep", "wide_deep"],
                                                 1, p=hp.layer_distribution).item(0))
                activation = str(np.random.choice([ 'tanh', 'linear', 'relu', 'selu',
                    'elu', 'sigmoid', 'hard_sigmoid', 'exponential', 'softmax',
                    'softplus', 'softsign', 'gaussian', 'sin', 'cos', 'swish'],
                                              1, p=hp.activation_distribution).item(0))
                d_out = None
                node_data["force_residual"] = random.random() < hp.p_force_residual
            node_data["activation"] = clean_activation(activation)
            node_data["node_type"] = node_type
            node_data['style'] = ""
            if node_type == 'sepconv1d':
                if d_out is None:
                    d_out = safe_sample(hp.min_filters, hp.max_filters)
                node_data["filters"] = d_out
                node_data["kernel_size"] = 1
            if node_type == "transformer":
                if d_out is None:
                    d_out = safe_sample(hp.min_units, hp.max_units) * hp.attn_heads
                node_data["d_model"] = d_out
                node_data["n_heads"] = 2 if d_out % 2 == 0 else 1
            if "k_conv" in node_type or node_type in ["deep", "wide_deep"]:
                layers = design_layers(hp, d_out, activation)
                if d_out is None:
                    d_out = layers[-1][0]
                node_data["stddev"] = hp.stddev
                node_data['layers'] = layers
                node_data["d_out"] = d_out
                if node_type in ["deep", "wide_deep"]:
                    node_data['kernel'] = node_type
                else:
                    node_data['kernel'] = "wide_deep" if random.random() < hp.p_wide_deep else "deep"
            label = f"{node_type}"
            log(f"set {node_id} to {label}")
            node_data["label"] = label
            node_data["color"] = "green"
            # we handle recurrent shapes:
            try:
                feedback_node_id = f"{node_id}_feedback"
                input_shape = (None, d_out)
                log(f"attempt to set input_shape for {feedback_node_id} to {input_shape}")
                feedback_node = G.node[feedback_node_id]
                feedback_node["input_shape"] = input_shape
                node_data["gives_feedback"] = True
            except Exception as e:
                log("ERROR HANDLING FEEDBACK SHAPE:", e)


def gaussian(x):
    return K.exp(-K.pow(x, 2))


def swish(x):
    return (K.sigmoid(x) * x)


def clean_activation(activation):
    log("clean_activation", activation)
    if activation == 'gaussian':
        return gaussian
    elif activation == 'swish':
        return swish
    elif activation == 'sin':
        return tf.math.sin
    elif activation == 'cos':
        return tf.math.cos
    else:
        return activation


def design_layers(hp, output_shape=None, act=None):
    n_layers = safe_sample(hp.min_k_layers, hp.max_k_layers)
    layers = []
    for layer_number in range(n_layers):
        units = safe_sample(hp.min_units, hp.max_units)
        if act is None:
            activation = np.random.choice(['tanh', 'linear', 'relu', 'selu', 'elu',
                                           'sigmoid', 'hard_sigmoid', 'exponential',
                                           'softmax', 'softplus', 'softsign',
                                           'gaussian', 'sin', 'cos', 'swish'],
                                          1, p=hp.activation_distribution)
            activation = clean_activation(str(activation.item(0)))
        else:
            activation = act
        layer = [units, activation]
        layers.append(layer)
    if output_shape is not None:
        layers[-1][0] = output_shape
    return layers


def make_model(hp):
    """Build the keras model described by a graph."""
    outputs = [get_output(id) for id in list(G.predecessors("sink"))]
    inputs = [G.node[id]['op'] for id in list(G.successors('source'))]
    outputs[0] = outputs[0] * hp.output_gain
    return K.Model(inputs, outputs), inputs, outputs


def get_output(id):
    """
    Get the output of a node in a computation graph.
    Pull inputs from predecessors.
    """
    global G
    node = G.node[id]
    keys = node.keys()
    node_type = node["node_type"]
    log('get output for', node)
    if node["output"] is not None:
        return node["output"]
    else:
        parent_ids = list(G.predecessors(id))
        if node_type is not "input":
            inputs = [get_output(parent_id) for parent_id in parent_ids]
            inputs = L.Concatenate()(inputs) if len(inputs) > 1 else inputs[0]
            if node_type is "recurrent":
                output = inputs
            else:
                op = build_op(id, inputs)
                log("got op", op)
                output = op(inputs)
                if "output_shape" not in keys and "gives_feedback" not in keys:
                    try:
                        output = L.Add()([inputs, output])
                    except Exception as e:
                        log("error adding inputs to output", e)
                    try:
                        output = L.Concatenate()([inputs, output])
                    except Exception as e:
                        log("error concatenating inputs to output", e)
        else:
            output = build_op(id)
        if "output_shape" not in keys:
            output = InstanceNormalization()(output)
        G.node[id]["output"] = output
        log("got output", output)
        return output


def build_op(id, inputs=None):
    """Make the keras operation to be executed at a given node."""
    global G
    node = G.node[id]
    log('build op for', node)
    node_type = node["node_type"]
    keys = node.keys()
    if "force_residual" in keys and "gives_feedback" not in keys:
        if node["force_residual"]:
            node["d_out"] = inputs.shape[-1]
    if node_type == "input":
        op = L.Input(node['input_shape'])
    if node_type == "sepconv1d":
        op = L.SeparableConv1D(node['filters'], node['kernel_size'], activation=node['activation'])
    if node_type in ["deep", "wide_deep"]:
        op = get_kernel(node_type, node["layers"], node["stddev"], inputs.shape[-1], node["d_out"], node["activation"], 0)
    if "k_conv" in node_type:
        op = KConvSet(node["kernel"], node["layers"], node["stddev"], inputs.shape[-1], node["d_out"], int(node_type[-1]))
    if node_type == "transformer":
        op = Transformer(node["d_model"], node["n_heads"])
    G.node[id]['op'] = op
    return op


def screenshot(G, i, j, k):
    """Make a png image of a graph."""
    log("SCREENSHOT", i, j, k, f"{G.order()} nodes, {G.size()} edges")
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update()
    A.draw(path=f"{IMAGE_PATH}/graph-{i}-{j}-{k}.png", prog="dot")


def show_model(model, i, j, k):
    if DEBUG:
        model.summary()
    K.utils.plot_model(model, f"archive/nets/{i}-{j}-{k}.png")


def make_gif(path):
    dirfiles = []
    for filename in os.listdir(path):
        log("parsed", filename)
        name, ext = os.path.splitext(filename)
        if name == 'model' or ext != ".png":
            continue
        log("convert", filename, "to jpg")
        cmd_string = f"convert '{IMAGE_PATH}/{name}.png' -gravity Center -thumbnail '{IMAGE_SIZE}' -extent {IMAGE_SIZE} -font Courier -gravity Center -undercolor '#00000080' -size {IMAGE_X}x{BRAND_Y} -fill white caption:'Bit Pharma' +swap -append -gravity Center '{IMAGE_PATH}/{name}.jpg'"
        os.system(cmd_string)
        dirfiles.append(f"{IMAGE_PATH}/{name}.jpg")
    dirfiles = natsorted(dirfiles)
    [log(dirfile) for dirfile in dirfiles]
    now = str(datetime.now()).replace(" ", "_")
    if MAKE_MOVIE:
        movie_path = f"archive/movies/net-{now}.mp4"
        writer = imageio.get_writer(movie_path, fps=(1/DURATION))
        [writer.append_data(imageio.imread(dirfile)) for dirfile in dirfiles]
        writer.close()
        log(f"SAVED MOVIE TO {movie_path}")
    else:
        images = [imageio.imread(dirfile) for dirfile in dirfiles]
        gif_path = f"archive/gifs/net-{now}.gif"
        imageio.mimsave(gif_path, images, duration=DURATION)
        log(f"SAVED GIF TO {gif_path}")


def space2dict(space):
    type_name = type(space).__name__
    space_dict = AttrDict({
        "object": space,
        "type_name": type_name,
    })
    if type_name == "Box":
        space_dict.shape = space.shape
        space_dict.dtype = tf.float32
        space_dict.low = space.low
        space_dict.high = space.high
    if type_name == "Discrete":
        space_dict.dtype = tf.int32
        space_dict.shape = (1)
        space_dict.high = space.n - 1
        space_dict.low = 0
    return space_dict


def prepare_tasks(tasks):
    tasks = [AttrDict(task) for task in tasks]
    prepared = []
    for task in tasks:
        if task.type is "env":
            env = gym.make(task.name)
            task.inputs = [space2dict(env.observation_space)]
            task.outputs = [space2dict(env.action_space)]
        prepared.append(task)
    del tasks
    return prepared


def get_agent(trial_number, hp, tasks):
    """Build a model given hyperparameters and input/output shapes."""
    global G

    layer_distribution = [hp.p_sepconv, hp.p_transformer, hp.p_k_conv1, hp.p_k_conv2, hp.p_k_conv3, hp.p_deep, hp.p_wide_deep]
    total = sum(layer_distribution)
    hp.layer_distribution = [x / total for x in layer_distribution]

    activation_distribution = [hp.p_tanh, hp.p_linear, hp.p_relu, hp.p_selu, hp.p_elu, hp.p_sigmoid, hp.p_hard_sigmoid, hp.p_exponential, hp.p_softmax, hp.p_softplus, hp.p_softsign, hp.p_gaussian, hp.p_sin, hp.p_cos, hp.p_swish]
    total = sum(activation_distribution)
    hp.activation_distribution = [x / total for x in activation_distribution]

    mutation_distribution = [hp.p_recurse, hp.p_decimate, hp.p_connect, hp.p_split_edges, hp.p_insert_motif, hp.p_add_edge, hp.p_delete_edge, hp.p_delete_node, hp.p_split_edge, hp.p_do_nothing]
    total = sum(mutation_distribution)
    hp.mutation_distribution = [x / total for x in mutation_distribution]

    [log(item) for item in hp.items()]
    prepared = prepare_tasks(tasks)
    G = get_initial_graph(prepared)

    screenshot(G, '0')
    recurse(hp)
    differentiate(hp)
    screenshot(G, hp.recursions + 1)
    model = make_model(hp)
    [log(item) for item in hp.items()]
    return model, str(datetime.now()).replace(" ", "_")

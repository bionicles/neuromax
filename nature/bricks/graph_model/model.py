# model.py - bion
# why?: recursively build keras models described as networkx graphs
import tensorflow_addons as tfa
import tensorflow as tf

from .bricks import Transformer, KConvSet, get_kernel
from helpers.get_path import get_path
from helpers.debug import log

K = tf.keras
B, L = K.backend, K.layers
InstanceNormalization = tfa.layers.InstanceNormalization


def show_model(model, folder, name, extension, debug=True):
    """summarize and plot a keras model"""
    if debug:
        model.summary()
    K.utils.plot_model(model, get_path(folder, name, extension))


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
                brick = build_brick(id, inputs)
                log("got brick", brick)
                output = brick(inputs)
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
            output = build_brick(id)
        if "output_shape" not in keys:
            output = InstanceNormalization()(output)
        G.node[id]["output"] = output
        log("got output", output)
        return output


# TODO: THIS SHOULD BE BUILD_BRICK!!! (WHAT'S A BRICK EXACTLY?)
def build_brick(id, inputs=None):
    """Make the keras operation to be executed at a given node."""
    global G
    node = G.node[id]
    log('build brick for', node)
    node_type = node["node_type"]
    keys = node.keys()
    # TODO: SIMPLIFY THE FUCK OUT OF THIS
    if "force_residual" in keys and "gives_feedback" not in keys:
        if node["force_residual"]:
            node["d_out"] = inputs.shape[-1]
    if node_type == "input":
        brick = L.Input(node['input_shape'])
    if node_type == "sepconv1d":
        brick = L.SeparableConv1D(node['filters'], node['kernel_size'], activation=node['activation'])
    if node_type in ["deep", "wide_deep"]:
        brick = get_kernel(node_type, node["layers"], node["stddev"], inputs.shape[-1], node["d_out"], node["activation"], 0)
    if "k_conv" in node_type:
        brick = KConvSet(node["kernel"], node["layers"], node["stddev"], inputs.shape[-1], node["d_out"], int(node_type[-1]))
    if node_type == "transformer":
        brick = Transformer(node["d_model"], node["n_heads"])
    G.node[id]['brick'] = brick
    return brick

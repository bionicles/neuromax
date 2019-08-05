# graphmodel.py - bion and kamel, july 2019
# why?: experiment faster with recursive neural architecture search

# more options for layers:
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
from functions.debug import log

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
    return model, hp

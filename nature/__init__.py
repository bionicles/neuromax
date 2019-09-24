# order matters!
from nature.bricks.optimizer import SGD, Adam, Nadam, Radam

from nature.bricks.built_in.regularize import L1L2, Regularizer
from nature.bricks.built_in.drop import Drop
from nature.bricks.chaos import EdgeOfChaos
from nature.bricks.built_in.initialize import Init
from nature.bricks.built_in.input import Input
from nature.bricks.built_in.conv_1D import Conv1D
from nature.bricks.built_in.conv_2D import Conv2D
from nature.bricks.built_in.deconv_2D import DConv2D
from nature.bricks.built_in.fc import FC
from nature.bricks.noisedrop import NoiseDrop
from nature.bricks.layer import Layer

from nature.bricks.norm import Norm
from nature.bricks.activations.linear import Linear
from nature.bricks.activations.logistic import Logistic
from nature.bricks.activations.polynomial import Polynomial
from nature.bricks.activations.pswish import PSwish, PolySwish
from nature.bricks.activations.logistic_map import LogisticMap
from nature.bricks.activations.fn import Fn
from nature.bricks.activations.no_op import NoOp
from nature.bricks.activations.channelwise import Channelwise
from nature.bricks.add_norm import AddNorm
from nature.bricks.norm_preact import NormPreact
from nature.bricks.quadratic import Quadratic
from nature.bricks.OP_1D import OP_1D
from nature.bricks.OP_FC import OP_FC
from nature.bricks.swag import SWAG
from nature.bricks.slim import Slim
# from nature.bricks.sandwich import Sandwich
# from nature.bricks.mixture import Mixture
from nature.bricks.attention import Attention

from nature.bricks.wide_deep import WideDeep
from nature.bricks.transformer import Transformer
from nature.bricks.recirculator import Recirculator
from nature.bricks.residual_block import ResBlock
from nature.bricks.dense_block import DenseBlock
from nature.bricks.resize import Resizer
from nature.bricks.merge import Merge
from nature.bricks.mlp import MLP
from nature.bricks.classifier import Classifier
from nature.bricks.coords import Coordinator
from nature.bricks.sensor import Sensor
from nature.bricks.ragged_actuator import RaggedActuator
from nature.bricks.image_actuator import ImageActuator
from nature.bricks.actuator import Actuator
from nature.bricks.stack import Stack
from nature.bricks.chain import Chain
from nature.bricks.brick import Brick

from nature.bricks.graph.helpers.count import count
from nature.bricks.graph.helpers.is_blacklisted import is_blacklisted
from nature.bricks.graph.helpers.screenshot_graph import screenshot_graph

from nature.bricks.graph.evolve.node import add_node, insert_node, insert_nodes
from nature.bricks.graph.evolve.edge import insert_edge, insert_edges
from nature.bricks.graph.evolve.regulon import Regulon
from nature.bricks.graph.evolve.motif import insert_motif, insert_motifs
from nature.bricks.graph.evolve.mutate import mutate

from nature.bricks.graph._output import get_output
from nature.bricks.graph._task import TaskModel

from nature.agent import Agent

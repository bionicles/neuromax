# order matters!

from nature.helpers.optimize import SGD, Adam, Nadam, Radam
from nature.helpers.regularize import L1L2
from nature.helpers.initialize import Init

from nature.bricks.conv import Conv1D, Conv2D, DConv2D
from nature.bricks.fc import FC
from nature.bricks.norm import Norm
from nature.bricks.linear import Linear
from nature.bricks.logistic import Logistic
from nature.bricks.fn import Fn
from nature.bricks.channelwise import Channelwise
from nature.bricks.pswish import PSwish
# from nature.bricks.multiply import Multiply
from nature.bricks.add import Add
from nature.bricks.norm_preact import NormPreact
from nature.bricks.quadratic import Quadratic
# from nature.bricks.sandwich import Sandwich
# from nature.bricks.mixture import Mixture
from nature.bricks.all_attn import AllAttention
from nature.bricks.layer import Layer
from nature.bricks.input import Input

from nature.bricks.residual_block import ResBlock
from nature.bricks.dense_block import DenseBlock
from nature.bricks.resize import Resizer
from nature.bricks.mlp import MLP
from nature.bricks.classifier import Classifier
from nature.bricks.coords_1D import ConcatCoords1D
from nature.bricks.coords_2D import ConcatCoords2D
from nature.bricks.coordinator import Coordinator
from nature.bricks.predictor import Predictor
from nature.bricks.ragged_actuator import RaggedActuator
from nature.bricks.image_actuator import ImageActuator
from nature.bricks.actuator import Actuator

from nature.bricks.graph.helpers.count import count
from nature.bricks.graph.helpers.add_node import add_node
from nature.bricks.graph.helpers.screenshot_graph import screenshot_graph

from nature.bricks.graph.evolve.regulon import Regulon
from nature.bricks.graph.evolve.insert import insert_motif, insert_motifs
from nature.bricks.graph.evolve.mutate import mutate

from nature.brick import Brick
from nature.bricks.graph._output import get_output
from nature.bricks.graph._task import TaskModel

from nature.agent import Agent

# order matters!

from nature.brick import Brick

from nature.bricks.helpers.regularize import L1L2
from nature.bricks.helpers.initialize import Init

from nature.bricks.layers.conv import Conv1D, Conv2D, DConv2D
from nature.bricks.layers.linear import Linear
from nature.bricks.layers.fn import Fn
from nature.bricks.layers.layer import Layer
from nature.bricks.layers.input import Input
from nature.bricks.layers.norm import Norm

from nature.bricks.resize import Resizer
from nature.bricks.residual_block import ResidualBlock
from nature.bricks.elemental import Multiply
from nature.bricks.dense_block import DenseBlock
from nature.bricks.norm_preact import NormPreact
from nature.bricks.mlp import MLP
from nature.bricks.swag import SWAG

from nature.bricks.interfaces.classifier import Classifier
from nature.bricks.interfaces.ragged_actuator import RaggedActuator
from nature.bricks.interfaces.ragged_sensor import RaggedSensor
from nature.bricks.interfaces.image_actuator import ImageActuator
from nature.bricks.interfaces.image_sensor import ImageSensor
from nature.bricks.interfaces.interface import Coder, Actuator


from nature.bricks.graph.helpers.count import count
from nature.bricks.graph.helpers.add_node import add_node
from nature.bricks.graph.helpers.screenshot_graph import screenshot_graph

from nature.bricks.graph.evolve.mutate import mutate
from nature.bricks.graph.evolve.insert import insert_motif, insert_motifs
from nature.bricks.graph.evolve.regulon import Regulon

from nature.bricks.graph._output import get_output
from nature.bricks.graph._brick import GraphBrick
from nature.bricks.graph._task import TaskModel


from nature.agent import Agent

# order matters!

from nature.bricks.helpers.regularize import get_l1_l2
from nature.bricks.helpers.initialize import get_init

from nature.bricks.layers.conv import use_conv_1D, use_conv_2D, use_deconv_2D
from nature.bricks.layers.linear import use_linear
from nature.bricks.layers.fn import use_fn
from nature.bricks.layers.layer import use_layer
from nature.bricks.layers.input import use_input
from nature.bricks.resize import use_resizer
from nature.bricks.layers.norm import use_norm

from nature.bricks.residual_block import use_residual_block
from nature.bricks.elemental import use_multiply
from nature.bricks.dense_block import use_dense_block
from nature.bricks.norm_preact import use_norm_preact
from nature.bricks.mlp import use_mlp
from nature.bricks.swag import use_swag
from nature.bricks.graph import Graph
from nature.bricks.graph_model import use_graph_model

from nature.bricks.interface.classify import add_classifier
from nature.bricks.interface.ragged_actuator import use_ragged_actuator
from nature.bricks.interface.ragged_sensor import use_ragged_sensor
from nature.bricks.interface.image_actuator import use_image_actuator
from nature.bricks.interface.image_sensor import use_image_sensor
from nature.bricks.interface.interface import use_coder, use_actuator

from nature.agent import Agent

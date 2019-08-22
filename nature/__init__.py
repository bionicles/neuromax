# order matters!

# helpers
from nature.bricks.helpers.regularize import get_l1_l2
from nature.bricks.helpers.chaos import get_chaos

# layers
from nature.bricks.layers.conv import use_conv_1D, use_conv_2D, use_deconv_2D
from nature.bricks.layers.fn import swish, clean_activation, use_fn
from nature.bricks.layers.layer import use_layer
from nature.bricks.layers.input import use_input
from nature.bricks.layers.dense import use_dense
from nature.bricks.layers.resize import use_flatten_resize_reshape
from nature.bricks.layers.attn import use_attn
from nature.bricks.layers.norm import use_norm

# bricks
from nature.bricks.residual_block import use_residual_block
from nature.bricks.elemental import use_add, use_multiply
from nature.bricks.dense_block import use_dense_block
from nature.bricks.mlp import use_mlp
from nature.bricks.swag import use_swag
from nature.bricks.graph_model.graph_model import use_graph_model

# interfaces
from nature.bricks.interface.image import use_image_sensor, use_image_actuator
from nature.bricks.interface.interface import use_interface

# higher-level stuff
from nature.brick import Brick
from nature.agent import Agent

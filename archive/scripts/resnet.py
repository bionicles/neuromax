# resnet.py
# why? define a resnet component for RL agents

from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras import Model

def make_layer(tuple):
    return Dense(units=tuple[0], activation=tuple[1])

def make_mlp(input, array):
    input
    for tuple in array:
        output = layer(tuple)
    return Model(input, output)

def make_resnet(in_shape, blocks):
    input = Input(shape=input_shape)
    first_block = make_mlp(blocks[0])
    output = first_block(input)
    prior = output
    for block in range(1, len(blocks)):
        mlp = make_mlp(block)
        output = mlp(output)
        output = Add([output, prior])
        prior = output
    return Model(input, output)

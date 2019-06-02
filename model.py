# model.py
# why? run the model on the input

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model


# we make one block in a res-net
def make_block(tensor, config):
    output = Dense(tensor, activation=config.ACTIVATION)
    for layer_num in config.NUM_LAYERS - 1:
        output = Dense(output, activation=config.ACTIVATION)
    return output


# we make a model
def make_model(config):
    input = Input(shape=(1, None, config.ATOM_DIMENSION))
    output = make_block(input, config)
    for i in range(config.NUM_BLOCKS):
        output += make_block(out, config)
    return Model(input, output)

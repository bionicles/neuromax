# model.py
# why? run the model on the input

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

def make_model(config):
    in = Input(shape=(1, None, config.ATOM_DIMENSION)
    out = make_block(in)
    for block_num in config.NUM_BLOCKS - 1:
        out = out + make_block(out)
    return Model(in, out)

def make_block(tensor, config):
    out = Dense(tensor)
    for layer_num in config.NUM_LAYERS - 1:
        out = Dense(out)
    return out

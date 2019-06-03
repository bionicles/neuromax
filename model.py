# model.py
# why? run the model on the input

from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model


# we make one block in a res-net
def make_block(input_layer, config):
    output = Dense(units = config.UNITS, activation=config.ACTIVATION)(input_layer)
    for layer_num in range(config.NUM_LAYERS - 1):
        output = Dense(units = config.UNITS, activation=config.ACTIVATION)(output)
        if config.DROPOUT: #True or False
        	output = Dropout(rate = config.DROPOUT_RATE)(output)
    return output


# we make a model
def make_model(config):
    input = Input(shape=(1, None, config.ATOM_DIMENSION))
    output = make_block(input, config)
    for i in range(config.NUM_BLOCKS):
        output += make_block(output, config)
    model_output = Dense(units = config.OUTPUT_SHAPE, activation = config.ACTIVATION)(output) 
    return Model(input, model_output)

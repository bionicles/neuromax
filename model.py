# model.py
# why? run the model on the input

from tensorflow.keras.layers import Input, Dense, Dropout, average, Layer
from tensorflow.keras import Model


# we make one block in a res-net
def make_block(input_layer, config, block_num):
    output = Dense(units = config.UNITS, activation=config.ACTIVATION)(input_layer)
    for layer_num in range(config.NUM_LAYERS - 1):
        output = Dense(units = config.UNITS, activation=config.ACTIVATION, name = 'block_{}'.format(block_num))(output)
        if config.DROPOUT: #True or False
        	output = Dropout(rate = config.DROPOUT_RATE)(output) # not sure if dropout layer support names option
    if block_num%config.CONNECT_BLOCK_AT == 0: #exp: each four separated blocks connect them through average/add/dot... layer
    	output = average([output, ' how to call layers by their name for averaging'])
    return output


# we make a model
def make_model(config):
    input = Input(shape=(1, None, config.ATOM_DIMENSION))
    output = make_block(input, config, block_num = 0) # initial block
    for i in range(1, config.NUM_BLOCKS + 1): #start count blocks from 1, useful to connect blocks using their numbers
        output = make_block(output, config, block_num)
    model_output = Dense(units = config.OUTPUT_SHAPE, activation = config.ACTIVATION)(output) 
    return Model(input, model_output)

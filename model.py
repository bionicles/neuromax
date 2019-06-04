# model.py
# why? run the model on the input

from tensorflow.keras.layers import Input, Dense, Dropout, average, Layer
from tensorflow.keras import Model
from keras.utils import print_summary
class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

# we keep parameters together to save time
config = AttrDict({
    "ACTIVATION": "tanh",
    "NUM_BLOCKS": 20,
    "NUM_LAYERS": 2,
    "UNITS": 50,
    "DROPOUT_RATE": 0.0,
    "CONNECT_BLOCK_AT": 4,
    "OUTPUT_SHAPE": 3,
    "DROPOUT": False,
    "DROPOUT_RATE": 0.0,
    "ATOM_DIMENSION": 17
})

# we make one block in a res-net
def make_block(input_layer, config, block_num):
    output = Dense(units = config.UNITS, activation=config.ACTIVATION)(input_layer)
    for layer_num in range(config.NUM_LAYERS - 1):
        output = Dense(units = config.UNITS, activation=config.ACTIVATION, name = 'block_{}'.format(block_num))(output)
        if config.DROPOUT: #True or False
        	output = Dropout(rate = config.DROPOUT_RATE)(output) # not sure if dropout layer support names option
    return output
# we make a model
def make_model(config):
    input = Input(shape=(1, None, config.ATOM_DIMENSION))
    output = make_block(input, config, block_num = 0) # initial block
    output_shortcut = output
    for i in range(1, config.NUM_BLOCKS): #start count blocks from 1, useful to connect blocks using their numbers
        output = make_block(output, config, i)
        if i%config.CONNECT_BLOCK_AT == 0:
            output = average([output, output_shortcut])
            output_shortcut = output
    model_output = Dense(units = config.OUTPUT_SHAPE, activation = config.ACTIVATION)(output) 
    return Model(input, model_output)

model = make_model(config)
print_summary(model)
# resnet.py
# why? define a resnet component for RL agents

from tensorflow.keras.layers import Input, Dense, Dropout, average, Layer
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model

# we make one block in a res-net
def make_block(input_layer, config, block_num):
    output = Dense(units = config.UNITS, activation=config.ACTIVATION)(input_layer)
    for layer_num in range(config.NUM_LAYERS - 1):
        output = Dense(units = config.UNITS, activation=config.ACTIVATION, name = 'block_{}'.format(block_num))(output)
        if config.DROPOUT: #True or False
        	output = Dropout(rate = config.DROPOUT_RATE)(output) # not sure if dropout layer support names option
    return output
# we make a model
def make_resnet(config):
    input = Input(shape=(config.ATOM_DIMENSION, ))
    output = make_block(input, config, block_num = 0) # initial block
    output_shortcut = output
    for i in range(1, config.NUM_BLOCKS): #start count blocks from 1, useful to connect blocks using their numbers
        output = make_block(output, config, i)
        if i%config.CONNECT_BLOCK_AT == 0:
            output = average([output, output_shortcut])
            output_shortcut = output
    model_output = Dense(units = config.OUTPUT_SHAPE, activation = config.ACTIVATION, name = "Output_Layer")(output)
    """model = Model(input, model_output)
    model.summary()
    try:
        plot_model(model, show_shapes = True)
    except:
        print("Fail to save the model architecture!!")"""
    return model_output

from tensorflow.keras.layers import Input, Attention, Concatenate, Activation, Dense, Dropout, Add
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.backend import random_normal
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras import Model
import tensorflow as tf
# https://ray.readthedocs.io/en/latest/rllib-models.html
# https://github.com/ray-project/ray/blob/master/python/ray/rllib/examples/custom_loss.py


class ResNet(Model):
    def __init__(self, name, in1, in2, layers, units, blocks, gain, l1, l2):
        self.name = name
        self.in1 = in1
        self.in2 = in2
        self.layers = layers
        self.units = units
        self.blocks = blocks
        self.gain = gain
        self.l1 = l1
        self.l2 = l2

    def make_block(self, features, noise):
        Attention_layer = Attention()([features, features])
        block_output = Concatenate(2)([Attention_layer, noise])
        block_output = Activation('tanh')(block_output)
        for layer_number in range(0, round(self.layers.item())-1):
            block_output = Dense(self.units,
                                 kernel_initializer=Orthogonal(self.gain),
                                 kernel_regularizer=L1L2(self.l1, self.l2),
                                 bias_regularizer=L1L2(self.l1, self.l2),
                                 activation='tanh'
                                 )(block_output)
            block_output = Dropout(0.5)(block_output)
        block_output = Dense(noise.shape[-1], 'tanh')(block_output)
        block_output = Add()([block_output, noise])
        return block_output

    def _build_layers_v2(self,  input_dict, num_outputs, options):
        features = Input((None, self.in1))
        noise = Input((None, self.in2))
        output = self.make_block(features, noise)
        for i in range(1, round(self.blocks.item())):
            output = self.make_block(features, output)
        output *= -1
        resnet = Model([features, noise], output)
        return resnet

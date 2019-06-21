from tensorflow.keras.layers import Layer, Dense, Concatenate
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras import Model, Input
import tensorflow as tf


def get_mlp(features, outputs, units_array):
    input = Input((features * 2))
    output = Dense(units, activation='tanh', kernel_initializer=Orthogonal)(input)
    for units in units_array[1:]:
        output = Dense(units, activation='tanh', kernel_initializer=Orthogonal)(output)
    output = Dense(outputs, activation='tanh', kernel_initializer=Orthogonal)(output)
    return Model(input, output)


class ConvPair(Layer):
    def __init__(self,
                 units_array=[2048, 2048],
                 features=16,
                 outputs=3):
        super(ConvPair, self).__init__()
        self.kernel = get_mlp(features, outputs, units_array) 

    def __call__(self, inputs):
        return tf.py_function(func=self.op, inp=inputs, Tout=tf.float32)

    def op(self, inputs):
        outputs = tf.zeros((inputs.shape[0], self.units_array[-1]))
        for i in num_atoms:
            for j in inputs:
                if i == j:
                    continue
                else:
                    k_in = tf.concat([inputs[i,:], inputs[j,:]])
                    outputs[i] = self.kernel(k_in)
        return outputs


def make_block(features, noise_or_output, n_layers):
    block_output = Concatenate(2)([features, noise_or_output])
    for layer_n in range(0, n_layers - 1):
        block_output = ConvPair()(block_output)
        block_output = Concatenate(2)([features, block_output])
    block_output = ConvPair()(block_output)
    block_output = Add()([block_output, noise_or_output])
    return block_output


def make_resnet(name, d_in, d_out, blocks, layers):
    features = Input((None, d_in))
    noise = Input((None, d_out))
    output = make_block(features, noise, layers)
    for i in range(1, round(blocks.item())):
        output = make_block(features, noise, layers)
    output *= -1
    return Model([features, noise], output)

import tensorflow.keras.layers as L
import tensorflow.keras as K
import tensorflow as tf

def make_block(features, MaybeNoiseOrOutput, layers, units, gain, l1, l2):
    Attention_layer = L.Attention()([features, features])
    block_output = L.Concatenate(2)([Attention_layer, MaybeNoiseOrOutput])
    block_output = L.Activation('tanh')(block_output)
    for layer_number in range(0, round(layers.item())-1):
        block_output = L.Dense(units,
                             kernel_initializer=K.initializers.Orthogonal(gain),
                             kernel_regularizer=K.regularizers.L1L2(l1, l2),
                             bias_regularizer=K.regularizers.L1L2(l1, l2),
                             activation='tanh'
                             )(block_output)
        block_output = L.Dropout(0.5)(block_output)
    block_output = L.Dense(MaybeNoiseOrOutput.shape[-1], 'tanh')(block_output)
    block_output = L.Add()([block_output, MaybeNoiseOrOutput])
    return block_output


def make_resnet(name, in1, in2, layers, units, blocks, gain, l1, l2):
    features = K.Input((None, in1))
    noise = K.Input((None, in2))
    output = make_block(features, noise, layers, units, gain, l1, l2)
    for i in range(1, round(blocks.item())):
        output = make_block(features, output, layers, units, gain, l1, l2)
    output *= -1
    resnet = K.Model([features, noise], output)
    return resnet

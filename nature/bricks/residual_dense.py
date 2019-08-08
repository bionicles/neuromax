import tensorflow as tf
K = tf.keras
L = K.layers
tfd = tfp.distributions
tfpl = tfp.layers
PREACTIVATION = 'relu'


def preact_conv2D(inputs, k=3, filters=64, tfp_layer=False):
    outputs = L.BatchNormalization()(inputs)
    outputs = L.Activation(PREACTIVATION)(outputs)
    if tfp_layer:
        outputs = tfpl.Convolution2DFlipout(filters, kernel_size=(k, k), padding='same')(outputs)
    else:
        outputs = L.Conv2D(filters, kernel_size=(k, k), padding='same')(outputs)
    return outputs


def ResidualBlock(inputs, kernal_size=3, filters=64):
    outputs = preact_conv2D(inputs, k=kernal_size, n_filters=filters)
    outputs = preact_conv2D(outputs, k=kernal_size, n_filters=filters)
    outputs = L.Add()([outputs, inputs])
    return outputs


def DenseBlock(stack, n_layers, growth_rate):
    new_features = []
    for i in range(n_layers):
        layer_out = preact_conv2D(stack, filters=growth_rate)
        new_features.append(layer_out)
        stack = tf.concat([stack, layer_out], axis=-1)
    new_features = tf.concat(new_features, axis=-1)
    return new_features


# Applying a stack of 5 Residual Blocks for a ResNet, just 5 lines of code
# If we wrote this out layer by layer, this would probably take 4-5x the number of lines
# x = ResidualBlock(x)
# x = ResidualBlock(x)
# x = ResidualBlock(x)
# x = ResidualBlock(x)
# x = ResidualBlock(x)

# Applying a stack of 5 Dense Blocks for a DenseNet, just 5 lines of code
# DenseNets are even more complex to implements than ResNets, so if we wrote
# this out layer by layer, this would probably take 5-10x the number of lines
# x = DenseBlock(x, n_layers=4, growth_rate=12)
# x = DenseBlock(x, n_layers=6, growth_rate=12)
# x = DenseBlock(x, n_layers=8, growth_rate=12)
# x = DenseBlock(x, n_layers=10, growth_rate=12)
# x = DenseBlock(x, n_layers=12, growth_rate=12)

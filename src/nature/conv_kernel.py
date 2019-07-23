# conv-kernel.py
# why?: build a resnet with kernel and attention set convolutions

import tensorflow as tf
# from make_dataset import load
# from pymol import cmd, util

B, L, K = tf.keras.backend, tf.keras.layers, tf.keras


class NoisyDropConnectDense(L.Dense):
    def __init__(self, *args, **kwargs):
        self.stddev = kwargs.pop('stddev')
        super(NoisyDropConnectDense, self).__init__(*args, **kwargs)

    @tf.function
    def add_noise(self):
        return (self.kernel + tf.random.truncated_normal(tf.shape(self.kernel), stddev=self.stddev),
                self.bias + tf.random.truncated_normal(tf.shape(self.bias), stddev=self.stddev))

    def call(self, x):
        kernel, bias = self.add_noise()
        return self.activation(tf.nn.bias_add(B.dot(x, tf.nn.dropout(kernel, 0.5)), bias))


def get_layer(units, hp):
    return NoisyDropConnectDense(units, activation="tanh", stddev=hp.stddev)


class KernelConvSet(L.Layer):
    def __init__(self, hp, d_features, d_output, N):
        super(KernelConvSet, self).__init__()
        self.kernel = get_kernel(hp.p, hp.p_layers, hp.p_units, hp, d_features, d_output, N)
        if N is 1:
            self.call = self.call_for_one
        elif N is 2:
            self.call = self.call_for_two
        elif N is 3:
            self.call = self.call_for_three

    def call_for_one(self, atoms):
        return tf.map_fn(lambda a1: self.kernel(a1), atoms)

    def call_for_two(self, atoms):
        return tf.map_fn(lambda a1: tf.reduce_sum(tf.map_fn(lambda a2: self.kernel([a1, a2]), atoms), axis=0), atoms)

    def call_for_three(self, atoms):
        return tf.map_fn(lambda a1: tf.reduce_sum(tf.map_fn(lambda a2: tf.map_fn(lambda a3: self.kernel([a1, a2, a3]), atoms), atoms), axis=0), atoms)


class ConvAttention(L.Layer):
    def __init__(self, d_features):
        super(ConvAttention, self).__init__()
        atom = K.Input((d_features,))
        atoms = K.Input((None, d_features,))
        output = L.Attention()(atom, atoms)
        self.attention = K.Model([atom, atoms], output)

    def call(self, atoms):
        return tf.map_fn(lambda atom: self.attention(atom, atoms), atoms)


def get_kernel(block_type, layers, units, hp, d_features, d_output, N):
    atom1 = K.Input((d_features, ))
    if N is 1:
        inputs = [atom1]
        concat = atom1
    elif N is 2:
        atom2 = K.Input((d_features, ))
        inputs = [atom1, atom2]
        d12 = L.Subtract()([atom1, atom2])
        concat = L.Concatenate()([d12, atom1, atom2])
    elif N is 3:
        atom2 = K.Input((d_features, ))
        atom3 = K.Input((d_features, ))
        inputs = [atom1, atom2, atom3]
        d12 = L.Subtract()([atom1, atom2])
        d13 = L.Subtract()([atom1, atom3])
        concat = L.Concatenate()([d12, d13, atom1, atom2, atom3])
    output = get_layer(units, hp)(concat)
    for layer in range(layers - 1):
        output = get_layer(units, hp)(output)
    if 'wide' in block_type:
        output = L.Concatenate(-1)([inputs, output])
    output = get_layer(d_output, hp)(output)
    return K.Model(inputs, output)


def get_block(block_type, hp, features, prior):
    if isinstance(prior, int):
        block_output = features
        d_output = prior
    else:
        block_output = L.Concatenate(-1)([features, prior])
        d_output = prior.shape[-1]
    d_features = block_output.shape[-1]
    block_output = ConvAttention(d_features)(block_output)  # convolutional attention
    block_output = KernelConvSet(hp, d_features, d_output, 3)(block_output)  # triplet convolution
    block_output = KernelConvSet(hp, d_features, d_output, 2)(block_output)  # pair convolution
    block_output = KernelConvSet(hp, d_features, d_output, 1)(block_output)  # atomic convolution
    if hp.norm is 'all':
        block_output = L.BatchNormalization()(block_output)
    if not isinstance(prior, int):
        block_output = L.Add()([prior, block_output])
    return block_output


def get_agent(trial_number, hp, d_in, d_out):
    print('\ntrial', trial_number, '\n')
    [print(f'   {k}={v}') for k, v in hp.items()]
    positions = K.Input((None, 3))
    features = K.Input((None, 7))
    stacked = L.Concatenate()([positions, features])
    normalized = L.BatchNormalization()(stacked) if hp.norm is not 'none' else stacked
    output = get_block(hp.p, hp, normalized, d_out)
    for i in range(hp.p_blocks - 1):
        output = get_block(hp.p, hp, normalized, output)
    trial_name = f'{trial_number}-' + '-'.join(f'{k}.{round(v, 4) if isinstance(v, float) else v}' for k, v in hp.items())
    return K.Model([positions, features], output, name=trial_name), trial_name

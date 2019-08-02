# conv-kernel.py
# why?: build a resnet with kernel and attention set convolutions

import tensorflow as tf
import random

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


def get_layer(units, activation, stddev):
    return NoisyDropConnectDense(units, activation="tanh", stddev=stddev)


def get_kernel(kernel_type, layers, stddev, d_in, d_out, last_activation, N, name=None):
    atom1 = K.Input((d_in, ))
    if N is 0:
        atom1 = K.Input((None, d_in))
        inputs = [atom1]
        concat = atom1
    elif N is 1:
        inputs = [atom1]
        concat = atom1
    elif N is 2:
        atom2 = K.Input((d_in, ))
        inputs = [atom1, atom2]
        d12 = L.Subtract()([atom1, atom2])
        concat = L.Concatenate(-1)([d12, atom1, atom2])
    elif N is 3:
        atom2 = K.Input((d_in, ))
        atom3 = K.Input((d_in, ))
        inputs = [atom1, atom2, atom3]
        d12 = L.Subtract()([atom1, atom2])
        d13 = L.Subtract()([atom1, atom3])
        concat = L.Concatenate(-1)([d12, d13, atom1, atom2, atom3])
    output = get_layer(layers[0][0], layers[0][1], stddev)(concat)
    for i in range(len(layers) - 1):
        output = get_layer(layers[i][0], layers[i][1], stddev)(output)
    if 'wide' in kernel_type:
        stuff_to_concat = inputs + [output]
        output = L.Concatenate(-1)(stuff_to_concat)
    output = get_layer(d_out, last_activation, stddev)(output)
    name = f"{len(layers)}-layer-{kernel_type}-{random.randint(0, 420)}" if name is None else name
    return K.Model(inputs, output, name=name)


class KConvSet(L.Layer):
    def __init__(self, kernel, layers, stddev, d_in, d_out, N):
        self.kernel = get_kernel(kernel, layers, stddev, d_in, d_out, "tanh", N)
        if N is 1:
            self.call = self.call_for_one
        elif N is 2:
            self.call = self.call_for_two
        elif N is 3:
            self.call = self.call_for_three
        super(KConvSet, self).__init__(name=f"KConvSet{N}-{len(layers)}-layer-{kernel}-{random.randint(0, 420)}")

    def call_for_one(self, atoms):
        return tf.map_fn(lambda a1: self.kernel(a1), atoms)

    def call_for_two(self, atoms):
        return tf.map_fn(lambda a1: tf.reduce_sum(tf.map_fn(lambda a2: self.kernel([a1, a2]), atoms), axis=0), atoms)

    def call_for_three(self, atoms):
        return tf.map_fn(lambda a1: tf.reduce_sum(tf.map_fn(lambda a2: tf.reduce_sum(tf.map_fn(lambda a3: self.kernel([a1, a2, a3]), atoms), axis=0), atoms), axis=0), atoms)

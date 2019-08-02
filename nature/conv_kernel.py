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

class Transformer(L.Layer):
    def __init__(self, d_model, n_heads):
        super(Transformer, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        assert d_model % self.n_heads == 0
        self.depth = d_model // self.n_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (n_heads, depth).
        Transpose the result such that the shape is (batch_size, n_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, atoms):
        batch_size = tf.shape(atoms)[0]

        q = self.wq(atoms)  # (batch_size, seq_len, d_model)
        k = self.wk(atoms)  # (batch_size, seq_len, d_model)
        v = self.wv(atoms)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, n_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, n_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, n_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, n_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, n_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, n_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        return self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)


def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so scores sum to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


# class Transformer(L.Layer):
#     def __init__(self, node_type, shape, units):
#         super(SelfAttention, self).__init__()
#         atoms = K.Input((None, shape))
#         key = L.Dense(units)(atoms)
#         value = L.Dense(units)(atoms)
#         key = L.LayerNormalization()(key)
#         value = L.LayerNormalization()(value)
#         if node_type == "luong":
#             output = L.Attention()([key, value])
#         else:
#             output = L.AdditiveAttention()([key, value])
#         output = L.LayerNormalization()(output)
#         output = L.Dense(units)(output)
#         self.attention = K.Model(atoms, output)
#
#     def call(self, inputs):
#         return self.attention([inputs, inputs])
#


# def get_block(block_type, hp, features, prior):
#     if isinstance(prior, int):
#         block_output = features
#         d_output = prior
#     else:
#         block_output = L.Concatenate(-1)([features, prior])
#         d_output = prior.shape[-1]
#     d_features = block_output.shape[-1]
#     block_output = SelfAttention(d_features)(block_output)  # convolutional attention
#     block_output = KernelConvSet(hp, d_features, d_output, 3)(block_output)  # triplet convolution
#     block_output = KernelConvSet(hp, d_features, d_output, 2)(block_output)  # pair convolution
#     block_output = KernelConvSet(hp, d_features, d_output, 1)(block_output)  # atomic convolution
#     if hp.norm is 'all':
#         block_output = L.BatchNormalization()(block_output)
#     if not isinstance(prior, int):
#         block_output = L.Add()([prior, block_output])
#     return block_output
#
#
# def get_agent(trial_number, hp, d_in, d_out):
#     print('\ntrial', trial_number, '\n')
#     [print(f'   {k}={v}') for k, v in hp.items()]
#     positions = K.Input((None, 3))
#     features = K.Input((None, 7))
#     stacked = L.Concatenate()([positions, features])
#     normalized = L.BatchNormalization()(stacked) if hp.norm is not 'none' else stacked
#     output = get_block(hp.p, hp, normalized, d_out)
#     for i in range(hp.p_blocks - 1):
#         output = get_block(hp.p, hp, normalized, output)
#     trial_name = f'{trial_number}-' + '-'.join(f'{k}.{round(v, 4) if isinstance(v, float) else v}' for k, v in hp.items())
#     return K.Model([positions, features], output, name=trial_name), trial_name

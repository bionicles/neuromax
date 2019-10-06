import tensorflow as tf
import nature


L = tf.keras.layers

INIT = nature.Init
REG = nature.L1L2
LAYER = nature.Layer
D_MODEL_OPTIONS = [8, 16, 32, 64, 128, 256]
MEMORY_SIZE_OPTIONS = [32, 512]
N_HEADS_OPTIONS = [1, 2, 4]
DROP_OPTIONS = [0., 0.5]
UNITS = None


class Attention(L.Layer):

    def __init__(self, AI, units=UNITS, layer_fn=LAYER):
        super(Attention, self).__init__()
        self.memory_size = AI.pull("attn_memory_size", MEMORY_SIZE_OPTIONS)
        self.d_model = AI.pull("attn_d_model", D_MODEL_OPTIONS)
        self.n_heads = AI.pull("attn_n_heads", N_HEADS_OPTIONS)
        self.p_drop = AI.pull("attn_p_drop", DROP_OPTIONS)
        assert self.d_model % self.n_heads == 0
        self.depth = self.d_model // self.n_heads
        self.delta = nature.Delta(AI)
        self.memory = self.add_weight(
            'memory', (1, self.memory_size, self.d_model), initializer=INIT(),
            regularizer=REG(), trainable=False)
        self.dense = nature.Layer(AI, units=self.d_model, layer_fn=layer_fn)
        self.wq = nature.Layer(AI, units=self.d_model, layer_fn=layer_fn)
        self.wk = nature.Layer(AI, units=self.d_model, layer_fn=layer_fn)
        self.wv = nature.Layer(AI, units=self.d_model, layer_fn=layer_fn)
        self.layer_fn = layer_fn
        self.units = units
        self.ai = AI

    def build(self, shape):
        units = self.units if self.units else shape[-1]
        self.channel_changer = tf.identity
        if units != self.d_model:
            self.channel_changer = nature.Layer(
                self.ai, units=units, layer_fn=self.layer_fn)
        super().build(shape)

    @tf.function
    def split_heads(self, x, batch_size):
        """Split the last dimension into (n_heads, depth).
        Transpose to (batch_size, n_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    @tf.function
    def call(self, sequence):
        shape = tf.shape(sequence)
        batch_size = shape[0]
        seq_len = shape[1]

        q = self.wq(sequence)  # (batch_size, seq_len, d_model)
        k = self.wk(sequence)  # (batch_size, seq_len, d_model)
        v = self.wv(sequence)  # (batch_size, seq_len, d_model)

        memory = tf.tile(self.memory, [batch_size, 1, 1])
        q = tf.concat([q, memory], 1)
        k = tf.concat([k, memory], 1)
        v = tf.concat([v, memory], 1)

        q = tf.nn.dropout(q, self.p_drop)
        k = tf.nn.dropout(k, self.p_drop)
        v = tf.nn.dropout(v, self.p_drop)

        q = self.split_heads(q, batch_size)  # (batch_size, n_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, n_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, n_heads, seq_len_v, depth)
        # scaled_attention.shape == (batch_size, n_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, n_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, n_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        attended = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        attended, memory = tf.split(attended, [seq_len, self.memory_size], axis=1)
        memory = tf.math.reduce_mean(memory, 0, keepdims=True)
        memory = self.delta(memory)
        self.memory.assign(memory)
        attended = self.channel_changer(attended)
        return attended


@tf.function
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

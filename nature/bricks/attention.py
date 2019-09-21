import tensorflow as tf
import nature

L = tf.keras.layers

LAYER = nature.Layer
MEMORY_SIZE = 512
D_MODEL = 8
N_HEADS = 2
P_DROP = 0.5
UNITS = None

class Attention(L.Layer):

    def __init__(
            self,
            memory_size=MEMORY_SIZE,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            units=UNITS,
            p_drop=P_DROP
        ):
        super(Attention, self).__init__()
        self.d_model, self.n_heads = d_model, n_heads
        assert d_model % self.n_heads == 0
        self.depth = d_model // self.n_heads
        self.dense = LAYER(units=d_model)
        self.wq = LAYER(units=d_model)
        self.wk = LAYER(units=d_model)
        self.wv = LAYER(units=d_model)
        self.memory = None
        if memory_size > 0:
            self.memory = self.add_weight(
                    shape=(1, memory_size, d_model), initializer=nature.Init(),
                    regularizer=nature.L1L2(), trainable=True)
        self.p_drop = p_drop
        self.units = units

    def build(self, shape):
        self.channel_changer = None
        if self.d_model is not shape[-1] or self.units is not None:
            units = self.units if self.units else shape[-1]
            self.channel_changer = LAYER(units=shape[-1])
        self.built = True

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

        if self.memory is not None:
            memory = tf.tile(self.memory, [batch_size, 1, 1])
            q = tf.concat([q, memory], 1)
            k = tf.concat([k, memory], 1)
            v = tf.concat([v, memory], 1)

        if self.p_drop > 0.:
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
        if self.memory is not None:
            attended, _ = tf.split(attended, [seq_len, MEMORY_SIZE], axis=1)
        if self.channel_changer is not None:
            attended = self.channel_changer(attended)
        return attended


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

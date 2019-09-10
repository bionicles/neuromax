import tensorflow as tf

from nature import use_linear

L = tf.keras.layers

D_MODEL = 32
N_HEADS = 2


def use_attn(parts):
    return MultiHeadAttention()


class MultiHeadAttention(L.Layer):

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS):
        super(MultiHeadAttention, self).__init__()
        self.d_model, self.n_heads = d_model, n_heads
        assert d_model % self.n_heads == 0
        self.depth = d_model // self.n_heads
        self.dense = use_linear(units=d_model)
        self.wq = use_linear(units=d_model)
        self.wk = use_linear(units=d_model)
        self.wv = use_linear(units=d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (n_heads, depth).
        Transpose to (batch_size, n_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, sequence):
        batch_size = tf.shape(sequence)[0]
        q = self.wq(sequence)  # (batch_size, seq_len, d_model)
        k = self.wk(sequence)  # (batch_size, seq_len, d_model)
        v = self.wv(sequence)  # (batch_size, seq_len, d_model)
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
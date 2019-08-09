import tensorflow as tf
import tensorflow_probability as tfp
L = tf.keras.layers
tfd = tfp.distributions
tfpl = tfp.layers

TFP_LAYER_OPTIONS = [tfpl.DenseReparameterization, tfpl.DenseFlipout]
TRUE_FALSE = [True, False]
D_MODEL_OPTIONS = [32, 64]
N_HEADS_OPTIONS = [1, 2]


class MultiHeadAttention(L.Layer):
    def __init__(self, agent, brick_id):
        super(MultiHeadAttention, self).__init__()
        self.pull_choices = agent.pull_choices
        self.pull_numbers = agent.pull_numbers
        self.agent = agent
        self.brick_id = brick_id
        d_model = self.pull_choices(f"{self.brick_id}_transformer_d_model",
                                    D_MODEL_OPTIONS)
        n_heads = self.pull_choices(f"{self.brick_id}_transformer_n_heads",
                                    N_HEADS_OPTIONS)
        use_tfp = self.pull_choices(f"{self.brick_id}_transformer_use_tfp",
                                    TRUE_FALSE)
        self.d_model, self.n_heads = d_model, n_heads
        assert d_model % self.n_heads == 0
        self.depth = d_model // self.n_heads
        if use_tfp:
            tfp_layer = self.pull_choices(f"{self.brick_id}_transformer_tfp_layer",
                                          TFP_LAYER_OPTIONS)
            self.wq = tfp_layer(d_model)
            self.wk = tfp_layer(d_model)
            self.wv = tfp_layer(d_model)
            self.dense = tfp_layer(d_model)
        else:
            self.wq = L.Dense(d_model)
            self.wk = L.Dense(d_model)
            self.wv = L.Dense(d_model)
            self.dense = L.Dense(d_model)

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

import tensorflow as tf

TWO_PI = 3.14159 * 2


def compute_entropy(x):
    stddev = tf.math.sqrt(tf.math.reduce_var(x))
    return 0.5 * tf.math.log(TWO_PI * tf.math.exp(stddev))

# def compute_entropy(x):
#     return x * tf.math.log(x)

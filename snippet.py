import tensorflow as tf

DTYPE = tf.float32

@tf.function(input_signature=[tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
                              tf.TensorSpec(shape=(None, None, 2), dtype=tf.float32),
                              tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                              tf.TensorSpec(shape=(None, None, 4), dtype=tf.float32),
                              tf.TensorSpec(shape=(None, None, 5), dtype=tf.float32),
                              tf.TensorSpec(shape=(None, None, 6), dtype=tf.float32),
                              tf.TensorSpec(shape=(None, None, 7), dtype=tf.float32),
                              tf.TensorSpec(shape=(None, None, 8), dtype=tf.float32),
                              tf.TensorSpec(shape=(None, None, 9), dtype=tf.float32)])
def print_useless_wall_of_text(one, two, three, four, five, six, seven, eight, nine):
    return one, two, three, four, five, six, seven, eight, nine


one = tf.random.normal((1, 123, 1), dtype=DTYPE)
two = tf.random.normal((1, 123, 2), dtype=DTYPE)
three = tf.random.normal((1, 123, 3), dtype=DTYPE)
four = tf.random.normal((1, 123, 4), dtype=DTYPE)
five = tf.random.normal((1, 123, 5), dtype=DTYPE)
six = tf.random.normal((1, 123, 6), dtype=DTYPE)
seven = tf.random.normal((1, 123, 7), dtype=DTYPE)
eight = tf.random.normal((1, 123, 8), dtype=DTYPE)
wrong = tf.random.normal((1, 123, 1), dtype=DTYPE)
stuff = print_useless_wall_of_text(one, two, three, four, five, six, seven, eight, wrong)

import tensorflow as tf
import time

dtype = tf.float16


# @tf.function
# def get_distances_rows(xyz):
#     print('tracing get_distances_rows')
#     xyz = tf.squeeze(xyz, 0)
#     return tf.map_fn(lambda atom_i: tf.reduce_sum(
#         tf.math.squared_difference(atom_i, xyz), 1), xyz)


@tf.function(input_signature=[tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32)])
def get_distances_matrix(xyz):
    xyz = tf.cast(tf.squeeze(xyz, 0), tf.float16)
    d = tf.reshape(tf.reduce_sum(xyz * xyz, 1), [-1, 1])
    return d - 2 * tf.matmul(xyz, xyz, transpose_b=True) + tf.transpose(d)


@tf.function
def get_losses(di, da):
    m = tf.math.square(tf.math.subtract(di, da))
    o = tf.math.exp(-da)
    zeros = tf.zeros(tf.shape(o)[0], dtype=dtype)
    o = tf.linalg.set_diag(o, zeros)
    m = tf.linalg.set_diag(m, zeros)
    return m, o


@tf.function
def test(N, method):
    a = tf.random.normal((1, N, 3))
    di = method(a)
    del a
    a = tf.random.normal((1, N, 3))
    m, o = get_losses(di, method(a))
    del a
    return m, o


N = 15000
while True:
    N = N + 100
    start = time.perf_counter()
    with tf.device('/gpu:0'):
        m, o = test(N, get_distances_matrix)
    print(N, time.perf_counter() - start, "seconds")

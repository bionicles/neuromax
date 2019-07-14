import tensorflow as tf

A = tf.random.normal((100, 10), mean=2.0)
B = tf.random.normal((10, 100), mean=2.0)
C = tf.abs(B-A)
print(C.shape)
# s, u, v = tf.linalg.svd(A)
# A_approx = tf.matmul(u, tf.matmul(tf.linalg.diag(s), v, adjoint_b=True))
# error = 100*tf.math.reduce_mean(A_approx-A)

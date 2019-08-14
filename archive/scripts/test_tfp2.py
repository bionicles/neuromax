import tensorflow as tf
import tensorflow_probability as tfp

d_out = 3

model = tf.keras.Sequential([
    tf.keras.Input((None, 10)),
    tfp.layers.Convolution1DFlipout(2 * d_out, 1)
])
for i in range(10000):
    features = tf.random.normal((1, i, 10))
    logits = model(features)
    print("entropy!:", dist.entropy())

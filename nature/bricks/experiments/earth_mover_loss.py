import tensorflow as tf

B = tf.keras.backend


@tf.function
def earth_mover_loss(y_true, y_pred):
    cdf_true = B.cumsum(y_true, axis=-1)
    cdf_pred = B.cumsum(y_pred, axis=-1)
    samplewise_emd = B.sqrt(
        B.mean(B.square(B.abs(cdf_true - cdf_pred)), axis=-1))
    return B.mean(samplewise_emd)

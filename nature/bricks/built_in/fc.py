import tensorflow as tf


from nature import Init, L1L2

K = tf.keras
L = K.layers
LAYER_CLASS = L.Dense  # NoiseDrop
INITIALIZER = Init
REGULARIZER = L1L2
ACTIVATION = None
UNITS = 2048


def FC(
        kernel_regularizer=REGULARIZER,
        activity_regularizer=REGULARIZER,
        bias_regularizer=REGULARIZER,
        kernel_initializer=INITIALIZER,
        bias_initializer=INITIALIZER,
        activation=ACTIVATION,
        units=UNITS,
        **kwargs
        ):
    return LAYER_CLASS(
            units=units,
            activation=activation,
            kernel_regularizer=kernel_regularizer(),
            # activity_regularizer=activity_regularizer(),
            bias_regularizer=bias_regularizer(),
            kernel_initializer=kernel_initializer(),
            bias_initializer=bias_initializer(dist='truncated')
            )

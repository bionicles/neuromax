from keras_radam.training import RAdamOptimizer
import tensorflow as tf

K = tf.keras

def SGD():
    return K.optimizers.SGD(
        learning_rate=3e-4, momentum=0.96, nesterov=True, clipvalue=0.01)

def Adam():
    return K.optimizers.Adam(learning_rate=3e-4, amsgrad=True, clipvalue=0.04)

def Nadam():
    return K.optimizers.Nadam(learning_rate=3e-4, clipvalue=0.04)


def Radam():
    return RAdamOptimizer(total_steps=9000, warmup_proportion=0.1, min_lr=1e-4)

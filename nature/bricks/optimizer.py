from keras_radam.training import RAdamOptimizer
import tensorflow as tf

K = tf.keras

WARMUP = False

def SGD():
    return K.optimizers.SGD(
        learning_rate=3e-4, momentum=0.96, nesterov=True, clipvalue=0.01)

def Adam():
    return K.optimizers.Adam(learning_rate=3e-4, amsgrad=True, clipvalue=0.04)

def Nadam():
    return K.optimizers.Nadam(learning_rate=3e-4, clipvalue=0.04)


def Radam():
    if WARMUP:
        return RAdamOptimizer(
            total_steps=999, warmup_proportion=0.04, min_lr=1e-6, amsgrad=True)
    else:
        return RAdamOptimizer(amsgrad=True)

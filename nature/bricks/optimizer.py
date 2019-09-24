from keras_radam.training import RAdamOptimizer
import tensorflow as tf


Opt = tf.keras.optimizers
CLIPVALUE = 0.2
WARMUP = False


def SGD():
    return Opt.SGD(
        learning_rate=3e-4, momentum=0.96, nesterov=True, clipvalue=0.01)


def Adam():
    return Opt.Adam(learning_rate=3e-4, amsgrad=True, clipvalue=0.04)


def Nadam():
    return Opt.Nadam(learning_rate=3e-4, clipvalue=0.04)


def Radam():
    if WARMUP:
        return RAdamOptimizer(
            total_steps=999, warmup_proportion=0.04, min_lr=1e-6, amsgrad=True)
    else:
        return RAdamOptimizer(amsgrad=True)

from keras_radam.training import RAdamOptimizer
# from tensorflow_addons.optimizers import Lookahead
import tensorflow as tf

Opt = tf.keras.optimizers
WARMUP = True
LOOK = True


def look(opt):
    # if not LOOK:
    return opt
    # return Lookahead(opt, sync_period=6, slow_step_size=0.5)


def SGD():
    opt = Opt.SGD(
        learning_rate=3e-4, momentum=0.96, nesterov=True, clipvalue=0.01)
    return look(opt)


def Adam():
    opt = Opt.Adam(learning_rate=3e-4, amsgrad=True, clipvalue=0.04)
    return look(opt)


def Nadam():
    opt = Opt.Nadam(learning_rate=3e-4, clipvalue=0.04)
    return look(opt)


def Radam():
    if WARMUP:
        opt = RAdamOptimizer(
            learning_rate=1e-3, min_lr=1e-5,
            total_steps=100000, warmup_proportion=0.01, amsgrad=True)
    else:
        opt = RAdamOptimizer(amsgrad=True)
    return look(opt)

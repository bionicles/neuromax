from keras_radam.training import RAdamOptimizer
import tensorflow as tf


Opt = tf.keras.optimizers


def wrap(opt):
    return opt
    # return MovingAverage(opt)
    # return Lookahead(opt, sync_period=6, slow_step_size=0.5)


def Radam(AI):
    steps = AI.pull("radam_steps", 1e4, 1e6, log_uniform=True)
    warmup = AI.pull("radam_warmup", 10, 1000)
    proportion = warmup / steps
    opt = RAdamOptimizer(
        learning_rate=1e-3, min_lr=1e-5,
        total_steps=steps, warmup_proportion=proportion, amsgrad=True)
    return wrap(opt)


# def SGD():
#     opt = Opt.SGD(
#         learning_rate=3e-4, momentum=0.96, nesterov=True, clipvalue=0.01)
#     return wrap(opt)
#
#
# def Adam():
#     opt = Opt.Adam(learning_rate=3e-4, amsgrad=True, clipvalue=0.04)
#     return wrap(opt)
#
#
# def Nadam():
#     opt = Opt.Nadam(learning_rate=3e-4, clipvalue=0.04)
#     return wrap(opt)
#
#
# def Adamw():
#     step = tf.Variable(0, trainable=False)
#     schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
#         [10000, 15000], [1e-0, 1e-1, 1e-2])
#     lr = 1e-1 * schedule(step)
#     opt = AdamW(learning_rate=lr, weight_decay=lambda: 1e-4 * schedule(step))
#     return wrap(opt)

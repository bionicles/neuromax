from tensorflow_addons.optimizers import MovingAverage, Lookahead, RectifiedAdam
import tensorflow as tf


Opt = tf.keras.optimizers


def wrap(opt):
    return MovingAverage(Lookahead(opt))


def Radam(AI):
    steps = AI.pull("radam_steps", 1e2, 1e5, log_uniform=True)
    return wrap(RectifiedAdam(total_steps=steps))


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

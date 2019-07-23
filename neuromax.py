# experiment.py: why? simplify
from plots import plot_objective, plot_evaluations
import matplotlib.pyplot as plt
from tensorboard import program
from attrdict import AttrDict
import numpy as np
import webbrowser
import shutil
import skopt
import math
import time
import gym
import os
from src.nature.nature import get_agent
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit=true'
import tensorflow as tf
# tracker = SummaryTracker()
plt.set_cmap("viridis")
B, L, K = tf.keras.backend, tf.keras.layers, tf.keras
# globals
qm9 = None
rxn = None
proteins = None
trial_number = 0
atom_index = 0
best_trial = ''
best_args = []
best = 12345678900
# search
ACQUISITION_FUNCTION = 'EIps'  # 'gp-hedge' if you don't care about speed
EPISODES_PER_TASK = 1000
N_EPOCHS = 1000
N_RANDOM_STARTS = 10
N_CALLS = 1000
# episodes
STOP_LOSS_MULTIPLE = 1.2
STARTING_TEMPERATURE = 273.15
FEATURE_DISTANCE_WEIGHT = 1.
STEP_DIVISOR = 10.
TENSORBOARD = False
MAX_STEPS = 420
# gif parameters
GIF_STYLE = "spheres"
IMAGE_SIZE = 256
N_MOVIES = 1
# agent
D_FEATURES = 7
D_OUT = 3
# hyperparameters
dimensions = [
    skopt.space.Integer(1, 4, name='recursions'),
    skopt.space.Real(0.04, 1, name='p_insert'),
    skopt.space.Integer(1, 2, name='min_layers'),
    skopt.space.Integer(2, 4, name='max_layers'),
    skopt.space.Integer(1, 2, name='min_nodes'),
    skopt.space.Integer(2, 4, name='max_nodes'),

    skopt.space.Integer(1, 15, name='min_filters'),
    skopt.space.Integer(16, 32, name='max_filters'),
    skopt.space.Categorical(['deep', 'wide_deep'], name='kernel_type'),
    skopt.space.Integer(1, 4, name='k_layers'),
    skopt.space.Integer(64, 127, name='min_units'),
    skopt.space.Integer(128, 512, name='max_units'),
    skopt.space.Real(0.001, 0.1, name='stddev'),

    skopt.space.Real(0.00001, 0.01, name='lr'),
    skopt.space.Integer(10, 10000000, name='decay')]


def trace_function(fn, *args):
    tf.summary.trace_on(graph=True, profiler=True)
    y = fn(*args)
    with writer.as_default():
        tf.summary.trace_export(name="trace_function", step=0, profiler_outdir=log_dir)
    return y


@skopt.utils.use_named_args(dimensions=dimensions)
def trial(**kwargs):
    global run_step, best, best_trial, best_args, trial_number, writer, ema, agent, optimizer, hp
    start_time = time.perf_counter()
    hp = AttrDict(kwargs)
    lr = tf.cast(hp.lr, tf.float32)

    agent, trial_name = get_agent(trial_number, hp, d_in=D_FEATURES, d_out=D_OUT)
    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    ema.apply(agent.weights)
    lr = tf.keras.experimental.CosineDecayRestarts(lr, hp.decay)
    optimizer = tf.keras.optimizers.Adam(lr, amsgrad=True)
    writer = tf.summary.create_file_writer(log_dir)

    def run_episodes(env):
        print("tracing train")
        changes = None
        episode = 0
        change = 0.
        while episode < EPISODES_PER_TASK and env.protein_number < env.proteins_in_dataset:
            current = env.reset()
            loss = 0.
            for step in tf.range(MAX_STEPS):
                with tf.GradientTape() as tape:
                    forces = agent(current)
                    current, loss, done, change = env.step(forces)
                gradients = tape.gradient(loss, agent.trainable_weights)
                optimizer.apply_gradients(zip(gradients, agent.trainable_weights))
                ema.apply(agent.weights)
            tf.print('episode', episode, 'with', change, "% change (lower is better)")
            tf.summary.scalar('change', change)
            if changes is not None:
                changes = tf.concat([changes, change], 0)
            else:
                changes = change
            episode = episode + 1
        return changes

    train = tf.function(run_episodes)
    try:
        with tf.device('/gpu:0'):
            changes = train(no_gif_env)
    except Exception as e:
        print("error in train step", e)

    median = tf.reduce_mean(changes)
    stddev = tf.reduce_var(changes)
    objective = median + stddev  # reward skill and consistency
    tf.print('changes:', changes)
    tf.print('median:', median, 'stddev:', stddev, 'objective:', objective)
    if tf.math.is_nan(objective):
        objective = 142.
    elif tf.math.less(objective, best):
        plot_path = os.path.join('.', 'runs', trial_name + '.png')
        K.utils.plot_model(agent, plot_path, show_shapes=True)
        agent.summary()
        averages = [ema.average(weight).numpy() for weight in agent.weights]
        agent.set_weights(averages)
        run_episodes()
        tf.saved_model.save(agent, os.path.join(log_dir, trial_name + ".h5"))
        best_trial = trial_name
        best = objective
    print('best_trial', best_trial)
    print('best', best)
    del agent, writer
    trial_number += 1
    if ACQUISITION_FUNCTION is 'EIps':
        elapsed = time.perf_counter() - start_time
        print(f'trial {trial_number} done in {elapsed}S')
        return objective, elapsed
    else:
        return objective


# class PlotCallback(object):
#     def __init__(self, path):
#         self.path = path
#
#     def __call__(self, results):
#         print(results)
#         if trial_number > 10:
#             _ = plot_evaluations(results)
#             plt.savefig(os.path.join(self.path, "evaluations.png"))
#             _ = plot_objective(results)
#             plt.savefig(os.path.join(self.path, "objective.png"))


def experiment():
    global log_dir, no_gif_env, gif_env
    log_dir = os.path.join('.', 'runs', str(time.time()))

    # if TENSORBOARD:
    #     tb = program.TensorBoard()
    #     tb.configure(argv=[None, '--logdir', log_dir])
    #     webbrowser.get(using='google-chrome').open(tb.launch()+'#scalars', new=2)

    # checkpoint_path = os.path.join(log_dir, 'checkpoint.pkl')
    # checkpointer = skopt.callbacks.CheckpointSaver(checkpoint_path, compress=9)
    # plotter = PlotCallback('.')
    # try:
    #     # res = skopt.load(checkpoint_path)
    #     # x0 = res.x_iters
    #     # y0 = res.func_vals
    #     results = skopt.gp_minimize(trial, dimensions, x0=x0, y0=y0, verbose=True, acq_func=ACQUISITION_FUNCTION, callback=[])
    # except Exception as e:
        # print(e)

    no_gif_env = gym.make("MolEnvNoGifs-v0")
    gif_env = gym.make("MolEnvGifs-v0")

    results = skopt.gp_minimize(trial, dimensions, verbose=True, acq_func=ACQUISITION_FUNCTION, callback=[])
    print(results)


if __name__ == '__main__':
    experiment()

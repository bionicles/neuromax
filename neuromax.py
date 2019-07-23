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
TEMPERATURE = 273.15
TENSORBOARD = False
MAX_STEPS = 420
# gif parameters
IMAGE_SIZE = 256
# agent
D_FEATURES = 7
D_OUT = 3
# hyperparameters
dimensions = [
    skopt.space.Integer(2, 3, name='recursions'),
    skopt.space.Real(0.04, 1, name='p_insert'),
    skopt.space.Integer(1, 2, name='min_layers'),
    skopt.space.Integer(2, 3, name='max_layers'),
    skopt.space.Integer(1, 2, name='min_nodes'),
    skopt.space.Integer(2, 3, name='max_nodes'),

    skopt.space.Integer(1, 4, name='min_filters'),
    skopt.space.Integer(8, 16, name='max_filters'),
    skopt.space.Categorical(['deep', 'wide_deep'], name='kernel_type'),
    skopt.space.Integer(1, 4, name='k_layers'),
    skopt.space.Integer(32, 127, name='min_units'),
    skopt.space.Integer(128, 256, name='max_units'),
    skopt.space.Real(0.001, 0.1, name='stddev'),

    skopt.space.Real(0.00001, 0.01, name='lr'),
    skopt.space.Integer(10, 10000000, name='decay')]


def trace_function(fn, *args):
    tf.summary.trace_on(graph=True, profiler=True)
    y = fn(*args)
    with writer.as_default():
        tf.summary.trace_export(name="trace_function", step=0, profiler_outdir=log_dir)
    return y


@tf.function
def parse_item(example):
    context_features = {'id': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = {'target': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'features': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'masses': tf.io.FixedLenSequenceFeature([], dtype=tf.string)}
    context, sequence = tf.io.parse_single_sequence_example(example, context_features=context_features, sequence_features=sequence_features)
    target = tf.reshape(tf.io.parse_tensor(sequence['target'][0], tf.float32), [-1, 10])
    positions = tf.reshape(tf.io.parse_tensor(sequence['positions'][0], tf.float32), [-1, 3])
    features = tf.reshape(tf.io.parse_tensor(sequence['features'][0], tf.float32), [-1, 7])
    masses = tf.reshape(tf.io.parse_tensor(sequence['masses'][0], tf.float32), [-1, 1])
    masses = tf.concat([masses, masses, masses], 1)
    n_atoms = tf.shape(positions)[0]
    id_string = context['id']
    return id_string, n_atoms, target, positions, features, masses


def read_shards(datatype):
    print("read_shards", datatype)
    dataset_path = os.path.join('.', 'src', 'nurture', 'mol', 'datasets', 'tfrecord', datatype)
    n_records = len(os.listdir(dataset_path))
    filenames = [os.path.join(dataset_path, str(i) + '.tfrecord') for i in range(n_records)]
    dataset = tf.data.TFRecordDataset(filenames, 'ZLIB')
    dataset = dataset.map(map_func=parse_item, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(1)
    return n_records, dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


@tf.function
def get_distances(a, b):  # L2
    a, b = tf.squeeze(a, 0), tf.squeeze(b, 0)
    return B.sum(B.square(tf.expand_dims(a, 0) - tf.expand_dims(b, 1)), axis=-1)


def get_loss(target_distances, current):
    # print("self.target", self.target)
    current = get_distances(current, current)
    # print("current", current)
    return tf.keras.losses.MAE(target_distances, current)


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

    @tf.function
    def run_episode(target, positions, features, masses):
        print("tracing run_episode")
        current = tf.concat([positions, features], -1)
        target_distances = get_distances(target, target)
        total_forces = tf.zeros_like(positions)
        velocities = tf.zeros_like(positions)
        forces = tf.zeros_like(positions)
        initial_loss = tf.reduce_sum(get_loss(target_distances, current))
        loss = 0.
        stop = 1.2 * initial_loss
        for step in tf.range(MAX_STEPS):
            with tf.GradientTape() as tape:
                forces = agent(current) / masses
                total_forces = total_forces + forces
                velocities = velocities + forces
                noise = tf.random.truncated_normal(tf.shape(positions), stddev=(0.001 * TEMPERATURE))
                positions = positions + velocities + noise
                current = tf.concat([positions, features], -1)
                loss = get_loss(target_distances, positions)
            gradients = tape.gradient([loss, tf.abs(total_forces), tf.abs(forces)], agent.trainable_weights)
            optimizer.apply_gradients(zip(gradients, agent.trainable_weights))
            ema.apply(agent.weights)
            loss = tf.reduce_sum(loss)
            tf.print(step, loss, stop)
            new_stop = loss * 1.2
            if step < 1:
                initial_loss = loss
                stop = new_stop
            if new_stop < stop:
                stop = new_stop
            elif step > 0 and (loss > stop or loss != loss):
                break
        change = ((loss - initial_loss) / initial_loss) * 100.
        return change if not tf.math.is_nan(change) else 420.

    @tf.function
    def train(dataset):
        print("tracing train")
        changes = 0.
        change = 0.
        for episode, (id_string, n_atoms, target, positions, features, masses) in dataset.enumerate():
            with tf.device('/gpu:0'):
                change = run_episode(target, positions, features, masses)
            tf.print('episode', episode + 1, 'protein', id_string,
                     'with', n_atoms, 'atoms',
                     change, "% change (lower is better)")
            tf.summary.scalar('change', change)
            if episode < 1:
                changes = tf.expand_dims(change, 0)
            elif episode == 1:
                changes = tf.stack(changes, tf.expand_dims(change))
            else:
                changes = tf.concat([changes, tf.expand_dims(change, 0)], 0)
            if episode >= EPISODES_PER_TASK:
                break
        return changes

    changes = train(proteins)

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
        # TODO: MAKE GIFS HERE
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
    global log_dir, proteins
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

    n_protein_records, proteins = read_shards("cif")

    results = skopt.gp_minimize(trial, dimensions, verbose=True, acq_func=ACQUISITION_FUNCTION, callback=[])
    print(results)


if __name__ == '__main__':
    experiment()

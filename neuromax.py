# experiment.py: why? simplify
from plots import plot_objective, plot_evaluations
import matplotlib.pyplot as plt
from tensorboard import program
from attrdict import AttrDict
import numpy as np
import webbrowser
import random
import imageio
import shutil
import skopt
import math
import time
import os
from src.nature.conv_kernel import get_agent
from src.nature.mol.make_dataset import read_shards
from src.nature.mol.mol import get_pdb_loss, get_work_loss
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
EPISODES_PER_DATASET = 1000
EPISODES_PER_TRIAL = 100
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
dimensions = [
    skopt.space.Integer(1, 4, name='p_blocks'),
    skopt.space.Categorical(['p_conv_deep', 'p_conv_wide_deep'], name='p'),
    skopt.space.Integer(2, 3, name='p_layers'),
    skopt.space.Integer(65, 1024, name='p_units'),
    skopt.space.Real(0.001, 0.1, name='stddev'),
    skopt.space.Categorical(['first', 'all'], name='norm'),
    skopt.space.Real(0.00001, 0.01, name='lr'),
    skopt.space.Integer(10, 10000000, name='decay')]
hyperpriors = [
    2,  # pair_blocks
    'p_conv_wide_deep',  # pair block type
    2,  # pair_layers
    4096,  # pair_units,
    0.02,  # stddev
    'all',  # norm
    0.004,  # lr
    10000000]  # decay_steps


def trace_function(fn, *args):
    tf.summary.trace_on(graph=True, profiler=True)
    y = fn(*args)
    with writer.as_default():
        tf.summary.trace_export(name="trace_function", step=0, profiler_outdir=log_dir)
    return y


@skopt.utils.use_named_args(dimensions=dimensions)
def trial(**kwargs):
    global run_step, best, best_trial, best_args, trial_number, writer, ema, agent, optimizer, hp, qm9, rxn, proteins
    start_time = time.perf_counter()
    hp = AttrDict(kwargs)
    lr = tf.cast(hp.lr, tf.float32)
    # try:
    #     strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
    #     print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))
    #     with strategy.scope():
    #         agent, trial_name = get_agent(trial_number, 16, 5, 3, hp)
    #         ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    #         averages = ema.apply(agent.weights)
    #         lr = tf.keras.experimental.CosineDecayRestarts(lr, hp.decay)
    #         optimizer = tf.keras.optimizers.Adam(lr, amsgrad=True)
    # except:
    agent, trial_name = get_agent(trial_number, hp, d_in=D_FEATURES, d_out=D_OUT)
    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    ema.apply(agent.weights)
    lr = tf.keras.experimental.CosineDecayRestarts(lr, hp.decay)
    optimizer = tf.keras.optimizers.Adam(lr, amsgrad=True)
    writer = tf.summary.create_file_writer(log_dir)

    @tf.function
    def run_episode(is_pdb, n_source_atoms, n_target_atoms, target_positions, positions, features, masses, target_numbers, numbers):
        print("tracing run_episode")
        total_forces = tf.zeros_like(positions)
        velocities = tf.zeros_like(positions)
        forces = tf.zeros_like(positions)
        initial_loss = 0.
        loss = 0.
        stop = 0.
        for step in tf.range(MAX_STEPS):
            with tf.GradientTape() as tape:
                forces = agent([positions, features]) / masses
                total_forces = total_forces + forces
                velocities = velocities + forces
                noise = tf.random.truncated_normal(tf.shape(positions), stddev=(0.001 * STARTING_TEMPERATURE * tf.math.exp(-1 * float(step) / STEP_DIVISOR)))
                positions = positions + velocities + noise
                try:
                    if is_pdb:
                        loss = get_pdb_loss(target_positions, positions, masses)
                    else:
                        loss = get_work_loss(n_source_atoms, n_target_atoms, target_positions, target_numbers, positions, numbers, masses, velocities)
                except Exception as e:
                    print("loss function failed", e)
                    return 100.
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
    def train(dataset, is_protein):
        print("tracing train")
        total_change = 0.
        change = 0.
        for episode, (type_string, id_string, n_source_atoms, n_target_atoms, target_positions, positions, features, masses, target_numbers, numbers) in dataset.enumerate():
            with tf.device('/gpu:0'):
                change = run_episode(is_protein, n_source_atoms, n_target_atoms, target_positions, positions, features, masses, target_numbers, numbers)
            tf.print(type_string, 'episode', episode, 'with',
                     n_source_atoms, 'source and',
                     n_target_atoms, 'target atoms',
                     change, "% change (lower is better)")
            tf.summary.scalar('change', change)
            total_change = total_change + change
            if episode >= EPISODES_PER_DATASET:
                break
        return total_change

    changes = []
    for epoch_number in range(N_EPOCHS):
        try:
            if epoch_number > 0:
                qm9 = qm9.shuffle(buffer_size=n_qm9_records)
                rxn = rxn.shuffle(buffer_size=n_rxn_records)
                proteins = proteins.shuffle(buffer_size=n_proteins)
            for dataset in [rxn]:
                is_protein = dataset == proteins
                total_change = train(dataset, is_protein)
            print('repeat', epoch_number + 1, 'total change:', total_change, '%')
            changes.append(total_change)
        except Exception as e:
            total_change = EPISODES_PER_TRIAL * 100 * STOP_LOSS_MULTIPLE
            print("error in epoch", epoch_number, e)

    median = np.median(changes)
    stddev = np.std(changes)
    objective = median + stddev  # reward skill and consistency
    print('changes:', changes)
    print('median:', median, 'stddev:', stddev, 'objective:', objective)
    if math.isnan(objective):
        objective = 100 * EPISODES_PER_TRIAL
    elif tf.math.less(objective, best):
        plot_path = os.path.join('.', 'runs', trial_name + '.png')
        K.utils.plot_model(agent, plot_path, show_shapes=True)
        agent.summary()
        # averages = [ema.average(weight).numpy() for weight in agent.weights]
        # agent.set_weights(averages)
        # make_gifs(agent)
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
    global log_dir, n_qm9_records, qm9, n_rxn_records, rxn, n_proteins, proteins
    log_dir = os.path.join('.', 'runs', str(time.time()))

    # if TENSORBOARD:
    #     tb = program.TensorBoard()
    #     tb.configure(argv=[None, '--logdir', log_dir])
    #     webbrowser.get(using='google-chrome').open(tb.launch()+'#scalars', new=2)

    n_qm9_records, qm9 = read_shards('xyz')
    n_rxn_records, rxn = read_shards('rxn')
    n_proteins, proteins = read_shards('cif')

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
    x0 = hyperpriors
    results = skopt.gp_minimize(trial, dimensions, x0=x0, verbose=True, acq_func=ACQUISITION_FUNCTION, callback=[])
    print(results)


if __name__ == '__main__':
    experiment()

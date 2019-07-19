# experiment.py: why? simplify
from plots import plot_objective, plot_evaluations
import matplotlib.pyplot as plt
from tensorboard import program
from attrdict import AttrDict
import numpy as np
import webbrowser
# from pympler.tracker import SummaryTracker
import random
import imageio
import shutil
import scipy
import skopt
import math
import time
import gc
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit=true'
import tensorflow as tf
# from make_dataset import load
# from pymol import cmd, util
# tracker = SummaryTracker()
plt.set_cmap("viridis")
B = tf.keras.backend
L = tf.keras.layers
K = tf.keras
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
    skopt.space.Integer(1, 8, name='p_blocks'),
    skopt.space.Categorical(['p_conv_deep', 'p_conv_wide_deep'], name='p'),
    skopt.space.Integer(1, 3, name='p_layers'),
    skopt.space.Integer(1, 1024, name='p_units'),
    skopt.space.Real(0.001, 0.1, name='stddev'),
    skopt.space.Categorical(['first', 'all'], name='norm'),
    skopt.space.Real(0.00001, 0.01, name='lr'),
    skopt.space.Integer(10, 10000000, name='decay')]  # major factor in step size
hyperpriors = [
    2,  # pair_blocks
    'p_conv_deep',  # pair block type
    2,  # pair_layers
    512,  # pair_units,
    0.02,  # stddev
    'first',  # norm
    0.004,  # lr
    10000000]  # decay_steps

# def show_memory_use():
#     all_objects = muppy.get_objects()
#     sum1 = summary.summarize(all_objects)
#     summary.print_(sum1)

def trace_function(fn, *args):
    tf.summary.trace_on(graph=True, profiler=True)
    y = fn(*args)
    with writer.as_default():
        tf.summary.trace_export(name="trace_function", step=0, profiler_outdir=log_dir)
    return y


class NoisyDropConnectDense(L.Dense):
    def __init__(self, *args, **kwargs):
        self.stddev = kwargs.pop('stddev')
        super(NoisyDropConnectDense, self).__init__(*args, **kwargs)

    # # @tf.function
    def add_noise(self):
        return (self.kernel + tf.random.truncated_normal(tf.shape(self.kernel), stddev=self.stddev),
                self.bias + tf.random.truncated_normal(tf.shape(self.bias), stddev=self.stddev))

    def call(self, x):
        kernel, bias = self.add_noise()
        return self.activation(tf.nn.bias_add(B.dot(x, tf.nn.dropout(kernel, 0.5)), bias))


def get_layer(units, hp):
    # return NoisyDropConnectDense(units, activation="tanh", stddev=hp.stddev)
    return L.Dense(units, activation="tanh")


class ConvPair(L.Layer):
    def __init__(self, hp, d_features, d_output):
        super(ConvPair, self).__init__()
        self.kernel = get_kernel(hp.p, hp.p_layers, hp.p_units, hp, d_features, d_output, pair=True)

    def call(self, inputs):
        return tf.map_fn(lambda a1: tf.reduce_sum(tf.map_fn(lambda a2: self.kernel([a1, a2]), inputs), axis=0), inputs)


def get_kernel(block_type, layers, units, hp, d_features, d_output, pair=False):
    atom1 = K.Input((d_features, ))
    atom2 = K.Input((d_features, ))
    inputs = L.Subtract()([atom1, atom2])
    output = get_layer(units, hp)(inputs)
    for layer in range(layers - 1):
        output = get_layer(units, hp)(output)
    if 'wide' in block_type:
        output = L.Concatenate(-1)([inputs, output])
    output = get_layer(d_output, hp)(output)
    return K.Model([atom1, atom2], output)


def get_block(block_type, hp, features, prior):
    if isinstance(prior, int):
        block_output = features
        d_output = prior
    else:
        block_output = L.Concatenate(-1)([features, prior])
        d_output = prior.shape[-1]
    d_features = block_output.shape[-1]
    if 'conv' in block_type:
        block_output = ConvPair(hp, d_features, d_output)(block_output)
    if hp.norm is 'all':
        block_output = L.BatchNormalization()(block_output)
    if not isinstance(prior, int):
        block_output = L.Add()([prior, block_output])
    return block_output


def get_agent(trial_number, hp, d_in, d_out):
    print('\ntrial', trial_number, '\n')
    [print(f'   {k}={v}') for k, v in hp.items()]
    positions = K.Input((None, 3))
    features = K.Input((None, 7))
    stacked = L.Concatenate()([positions, features])
    normalized = L.BatchNormalization()(stacked) if hp.norm is not 'none' else stacked
    output = get_block(hp.p, hp, normalized, d_out)
    for i in range(hp.p_blocks - 1):
        output = get_block(hp.p, hp, normalized, output)
    trial_name = f'{trial_number}-' + '-'.join(f'{k}.{round(v, 4) if isinstance(v, float) else v}' for k, v in hp.items())
    return K.Model([positions, features], output, name=trial_name), trial_name


@tf.function
def parse_item(example):
    context_features = {  # 'quantum_target': tf.io.FixedLenFeature([], dtype=tf.string),
                        'type': tf.io.FixedLenFeature([], dtype=tf.string),
                        'id': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = {'target_positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'target_numbers': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'features': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'numbers': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'masses': tf.io.FixedLenSequenceFeature([], dtype=tf.string)}
    context, sequence = tf.io.parse_single_sequence_example(example, context_features=context_features, sequence_features=sequence_features)
    target_positions = tf.reshape(tf.io.parse_tensor(sequence['target_positions'][0], tf.float32), [-1, 3])
    target_numbers = tf.reshape(tf.io.parse_tensor(sequence['target_numbers'][0], tf.float32), [-1, 1])
    positions = tf.reshape(tf.io.parse_tensor(sequence['positions'][0], tf.float32), [-1, 3])
    features = tf.reshape(tf.io.parse_tensor(sequence['features'][0], tf.float32), [-1, 7])
    numbers = tf.reshape(tf.io.parse_tensor(sequence['numbers'][0], tf.float32), [-1, 1])
    masses = tf.reshape(tf.io.parse_tensor(sequence['masses'][0], tf.float32), [-1, 1])
    # quantum_target = tf.io.parse_tensor(context['quantum_target'], tf.float32)
    masses = tf.concat([masses, masses, masses], 1)
    n_source_atoms = tf.shape(positions)[0]
    n_target_atoms = tf.shape(target_positions)[0]
    type_string = context['type']
    id_string = context['id']
    return type_string, id_string, n_source_atoms, n_target_atoms, target_positions, positions, features, masses, target_numbers, numbers


def read_shards(datatype):
    print("read_shards", datatype)
    dataset_path = os.path.join('.', 'datasets', 'tfrecord', datatype)
    n_records = len(os.listdir(dataset_path))
    filenames = [os.path.join(dataset_path, str(i) + '.tfrecord') for i in range(n_records)]
    dataset = tf.data.TFRecordDataset(filenames, 'ZLIB')
    dataset = dataset.map(map_func=parse_item, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(1)
    return n_records, dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

#
# def move_atom(xyz):
#     global atom_index
#     xyz = xyz.tolist()
#     cmd.translate(xyz, f'id {str(atom_index)}')
#     atom_index += 1
#
#
# def move_atoms(velocities):
#     global atom_index
#     atom_index = 0
#     np.array([move_atom(xyz) for xyz in velocities])
#
#
# def prepare_pymol():
#     cmd.show(GIF_STYLE)
#     cmd.unpick()
#     util.cbc()
#
#
# def make_gif(pdb_name, trial_name, pngs_path):
#     gif_name = f'{pdb_name}-{trial_name}.gif'
#     gif_path = os.path.join(".", "gifs", gif_name)
#     imagepaths, images = [], []
#     for stackname in os.listdir(pngs_path):
#         print('processing', stackname)
#         filepath = os.path.join(pngs_path, stackname)
#         imagepaths.append(filepath)
#     imagepaths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
#     for imagepath in imagepaths:
#         image = imageio.imread(imagepath)
#         images.append(image)
#     print('saving gif to', gif_path)
#     imageio.mimsave(gif_path + pdb_name + ".gif", images)


@tf.function
def igsp(source_positions, source_numbers, target_positions, target_numbers):
    print("tracing igsp")
    source_positions, source_numbers, target_positions, target_numbers = [tf.squeeze(x, 0) for x in [source_positions, source_numbers, target_positions, target_numbers]]
    columns, rows = tf.range(tf.shape(target_positions)[0]), tf.range(tf.shape(target_positions)[0])
    source_centroid = tf.reduce_mean(source_positions, axis=0)
    target_centroid = tf.reduce_mean(target_positions, axis=0)
    source_positions_copy = tf.identity(source_positions) - source_centroid
    target_positions_copy = tf.identity(target_positions) - target_centroid
    translation = target_centroid - source_centroid
    feature_distances = get_distances(source_numbers, target_numbers)
    rotation = tf.ones((3,3))
    for igsp_step in range(3):
        euclidean_distances = get_distances(source_positions_copy, target_positions_copy)
        compound_distances = euclidean_distances + 10. * feature_distances
        rows, columns = scipy.optimize.linear_sum_assignment(compound_distances)
        rows = tf.convert_to_tensor(rows, tf.int32)
        columns = tf.convert_to_tensor(columns, tf.int32)
        ordered_source_positions = tf.gather(source_positions_copy, columns)
        ordered_target_positions = tf.gather(target_positions_copy, rows)
        covariance = B.dot(tf.transpose(ordered_source_positions), ordered_target_positions)
        s, u, v = tf.linalg.svd(covariance)
        d = tf.linalg.det(v * tf.transpose(u))
        temporary_rotation = v * tf.linalg.diag([1., 1., d]) * tf.transpose(u)
        source_positions_copy = tf.tensordot(source_positions_copy, temporary_rotation, 1)
        rotation = temporary_rotation * rotation
        print('rotation', rotation)
    return rows, columns, translation, rotation


@tf.function
def get_work_loss(n_source_atoms, n_target_atoms, target_positions, target_numbers, positions, numbers, masses, velocities):
    print("tracing get_work_loss")
    rows, columns, translation, rotation = tf.numpy_function(igsp, [positions, numbers, target_positions, target_numbers], [tf.int32, tf.int32, tf.float32, tf.float32])
    aligned = tf.linalg.matmul((positions + translation), rotation)
    aligned = positions + translation
    gathered_positions = tf.squeeze(tf.gather(aligned, columns, axis=1), 0)
    gathered_target = tf.squeeze(tf.gather(target_positions, rows, axis=1), 0)
    distances = tf.linalg.diag_part(get_distances(gathered_positions, gathered_target))
    gathered_velocities = tf.reduce_sum( tf.squeeze(tf.gather(velocities, columns, axis=1), 0), -1)
    gathered_masses, _, _ = tf.split(tf.squeeze(tf.gather(masses, columns, axis=1), 0), 3, axis=-1)
    gathered_masses = tf.squeeze(gathered_masses, -1)
    work = 2. * distances - gathered_velocities
    tf.print(tf.shape(work))
    work = tf.multiply(gathered_masses, work)
    tf.print(tf.shape(work))
    print(work, type(work), tf.shape(work))
    tf.print(tf.shape(work))
    return work


@tf.function
def get_meta_loss(target_distances, positions):
    print("tracing get_meta_loss")
    positions = tf.squeeze(positions, 0)
    return K.losses.MAE(target_distances, get_distances(positions, positions))


@tf.function
def get_distances(a, b):
    print("tracing get_distances")
    return tf.reduce_sum(tf.math.abs(tf.expand_dims(a, 0) - tf.expand_dims(b, 1)), axis=-1)  # N, N, 1


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

    # @tf.function
    def run_episode(is_rxn, n_source_atoms, n_target_atoms, target_positions, positions, features, masses, target_numbers, numbers):
        print("tracing run_episode")
        if not is_rxn:
            target_distances = get_distances(tf.squeeze(target_positions, 0), tf.squeeze(target_positions, 0))
        total_forces = tf.zeros_like(positions)
        velocities = tf.zeros_like(positions)
        forces = tf.zeros_like(positions)
        initial_loss = 0.
        loss = 0.
        stop = 0.
        ones = tf.ones_like(positions)
        for step in tf.range(MAX_STEPS):
            with tf.GradientTape() as tape:
                forces = agent([positions, features]) / masses
                total_forces = total_forces + forces
                velocities = velocities + forces
                noise = tf.random.truncated_normal(tf.shape(positions), stddev=(0.001 * STARTING_TEMPERATURE * tf.math.exp(-1 * float(step) / STEP_DIVISOR)))
                positions = positions + velocities + noise
                # if not is_rxn:
                #     loss = get_meta_loss(target_distances, positions)
                # elif is_rxn:
                loss = get_work_loss(n_source_atoms, n_target_atoms, target_positions, target_numbers, positions, numbers, masses, velocities)
            gradients = tape.gradient([loss, tf.math.abs(total_forces), tf.math.abs(forces)], agent.trainable_weights)
            optimizer.apply_gradients(zip(gradients, agent.trainable_weights))
            ema.apply(agent.weights)
            loss = tf.reduce_sum(loss)
            # print(loss)
            new_stop = loss * 1.2
            if step < 1:
                initial_loss = loss
            if new_stop < stop or stop == 0.:
                stop = new_stop
            elif step > 0 and (loss > stop or loss != loss):
                break
        change = ((loss - initial_loss) / initial_loss)
        return change if not tf.math.is_nan(change) else 420.

    # @tf.function
    def train(dataset, is_rxn):
        print("tracing train")
        total_change = 0.
        change = 0.
        for episode, (type_string, id_string, n_source_atoms, n_target_atoms, target_positions, positions, features, masses, target_numbers, numbers) in dataset.enumerate():
            tf.print(episode, id_string, type_string, n_source_atoms, n_target_atoms)
            [tf.print(tf.shape(x)) for x in [target_positions, positions, features, masses, target_numbers, numbers]]
            with tf.device('/gpu:0'):
                change = run_episode(is_rxn, n_source_atoms, n_target_atoms, target_positions, positions, features, masses, target_numbers, numbers)
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
                # qm9 = qm9.shuffle(buffer_size=n_qm9_records)
                rxn = rxn.shuffle(buffer_size=n_rxn_records)
                # proteins = proteins.shuffle(buffer_size=n_proteins)
            for dataset in [rxn]:
                is_rxn = dataset == rxn
                total_change = train(dataset, is_rxn)
            # total_change = total_change.numpy().item(0)
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

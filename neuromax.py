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
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit=true'
import tensorflow as tf
from make_dataset import load
from pymol import cmd, util
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
# parameters
ACQUISITION_FUNCTION = 'EIps'  # 'gp-hedge' if you don't care about speed
STOP_LOSS_MULTIPLE = 1.2
EPISODES_PER_DATASET = 2
EPISODES_PER_TRIAL = 2
REPEATS_PER_TRIAL = 2
D_FEATURES = 7
D_COMPRESSED = 5
D_OUT = 3
TENSORBOARD = False
MAX_STEPS = 420
N_RANDOM_STARTS = 10
N_CALLS = 1000
GIF_STYLE = "spheres"
IMAGE_SIZE = 256
N_MOVIES = 1
# hyperparameters
dimensions = [
    skopt.space.Integer(1, 4, name='c_blocks'),
    skopt.space.Categorical(['c_res_deep', 'c_res_wide_deep', 'c_conv_deep', 'c_conv_wide_deep'], name="c"),
    skopt.space.Integer(1, 3, name='c_layers'),
    skopt.space.Integer(1, 1024, name='c_units'),
    skopt.space.Integer(1, 8, name='p_blocks'),
    skopt.space.Categorical(['p_res_deep', 'p_res_wide_deep', 'p_conv_deep', 'p_conv_wide_deep'], name='p'),
    skopt.space.Integer(1, 3, name='p_layers'),
    skopt.space.Integer(1, 1024, name='p_units'),
    skopt.space.Real(0.001, 0.1, name='stddev'),
    skopt.space.Categorical(['normal', 'noise', 'drop', 'noisedrop'], name='layer'),
    skopt.space.Categorical(['first', 'all'], name='norm'),
    skopt.space.Real(0.00001, 0.01, name='lr'),
    skopt.space.Integer(10, 10000000, name='decay')]  # major factor in step size
hyperpriors = [
    2,  # compressor_blocks,
    'c_conv_wide_deep',  # compressor block type
    2,  # compressor_layers
    512,  # compressor_units
    2,  # pair_blocks
    'p_conv_wide_deep',  # pair block type
    2,  # pair_layers
    512,  # pair_units,
    0.02,  # stddev
    'normal',
    'all',  # norm
    0.004,  # lr
    10000000]  # decay_steps


def trace_function(fn, *args):
    tf.summary.trace_on(graph=True, profiler=True)
    y = fn(*args)
    with writer.as_default():
        tf.summary.trace_export(name="trace_function", step=0, profiler_outdir=log_dir)
    return y


class NoisyDropConnectDense(L.Dense):
    def __init__(self, *args, **kwargs):
        layer = kwargs.pop('layer')
        self.stddev = kwargs.pop('stddev')
        super(NoisyDropConnectDense, self).__init__(*args, **kwargs)
        if layer is 'noisedrop':
            self.call = self.call_noise_drop
        elif layer is 'drop':
            self.call = self.call_drop
        elif layer is 'noise':
            self.call = self.call_noise
        else:
            self.call = self.normal_call

    def add_noise(self):
        return (self.kernel + tf.random.truncated_normal(tf.shape(self.kernel), stddev=self.stddev),
                self.bias + tf.random.truncated_normal(tf.shape(self.bias), stddev=self.stddev))

    def call_noise_drop(self, x):
        kernel, bias = self.add_noise()
        return self.activation(tf.nn.bias_add(B.dot(x, tf.nn.dropout(kernel, 0.5)), bias))

    def call_drop(self, x):
        return self.activation(tf.nn.bias_add(B.dot(x, tf.nn.dropout(self.kernel, 0.5)), self.bias))

    def call_noise(self, x):
        kernel, bias = self.add_noise()
        return self.activation(tf.nn.bias_add(B.dot(x, kernel), bias))

    def normal_call(self, x):
        return self.activation(tf.nn.bias_add(B.dot(x, self.kernel), self.bias))


def get_layer(units, hp):
    return NoisyDropConnectDense(units, activation="tanh", layer=hp.layer, stddev=hp.stddev)


class ConvKernel(L.Layer):
    def __init__(self, hp, d_features, d_output):
        super(ConvKernel, self).__init__()
        self.kernel = get_kernel(hp.c, hp.c_layers, hp.c_units, hp, d_features, d_output)

    def call(self, inputs):
        return tf.map_fn(self.kernel, inputs)


class ConvPair(L.Layer):
    def __init__(self, hp, d_features, d_output):
        super(ConvPair, self).__init__()
        self.kernel = get_kernel(hp.p, hp.p_layers, hp.p_units, hp, d_features, d_output, pair=True)

    def call(self, inputs):
        return tf.map_fn(lambda a1: tf.reduce_sum(tf.map_fn(lambda a2: self.kernel([a1, a2]), inputs), axis=0), inputs)


def get_kernel(block_type, layers, units, hp, d_features, d_output, pair=False):
    print(f"get {block_type} layers {layers} units {units} features{d_features} output{d_output}")
    input = K.Input((d_features, ))
    if pair:
        input2 = K.Input((d_features, ))
        arr1 = L.Lambda(lambda x: tf.split(x, d_features, -1))(input)
        arr2 = L.Lambda(lambda x: tf.split(x, d_features, -1))(input2)
        xyz1 = L.Concatenate()(arr1[0:3])
        xyz2 = L.Concatenate()(arr2[0:3])
        f1 = L.Concatenate()(arr1[3:])
        f2 = L.Concatenate()(arr2[3:])
        dxyz = L.Subtract()([xyz1, xyz2])
        inputs = L.Concatenate()([dxyz, f1, f2])
        output = get_layer(units, hp)(inputs)
    else:
        output = get_layer(units, hp)(input)
    for layer in range(layers - 1):
        output = get_layer(units, hp)(output)
    if 'wide' in block_type:
        output = L.Concatenate(-1)([input, output])
    output = get_layer(d_output, hp)(output)
    return K.Model([input, input2], output) if pair else K.Model(input, output)


def get_block(block_type, hp, features, prior):
    if isinstance(prior, int):
        print("\nprior is an int", prior)
        block_output = features
        d_output = prior
    else:
        print("\nprior is a tensor", prior)
        block_output = L.Concatenate(-1)([features, prior])
        d_output = prior.shape[-1]
    d_features = block_output.shape[-1]
    if block_type in ["c_conv_deep", "c_conv_wide_deep"]:
        block_output = ConvKernel(hp, d_features, d_output)(block_output)
    elif block_type in ["p_conv_deep", 'p_conv_wide_deep']:
        block_output = ConvPair(hp, d_features, d_output)(block_output)
    elif block_type in ["c_res_deep", "c_res_wide_deep"]:
        block_output = get_kernel(block_type, hp.c_layers, hp.c_units, hp, d_features, d_output)(block_output)
    elif block_type in ["p_res_deep", "p_res_wide_deep"]:
        block_output = get_kernel(block_type, hp.p_layers, hp.p_units, hp, d_features, d_output)(block_output)
    if hp.norm is 'all':
        print("block output pre norm", block_output)
        block_output = L.BatchNormalization()(block_output)
    if not isinstance(prior, int):
        block_output = L.Add()([prior, block_output])
    print('block output', block_output)
    return block_output


def get_agent(trial_number, hp, d_in, d_compressed, d_out):
    print('\ntrial', trial_number, '\n')
    [print(f'   {k}={v}') for k, v in hp.items()]
    positions = K.Input((None, 3))
    features = K.Input((None, 7))
    normalized_features = L.BatchNormalization()(features) if hp.norm is not 'none' else features
    compressed_features = get_block(hp.c, hp, normalized_features, d_compressed)
    for compressor_block in range(hp.c_blocks - 1):
        compressed_features = get_block(hp.c, hp, normalized_features, compressed_features)
    normalized_positions = L.BatchNormalization()(positions) if hp.norm is not 'none' else positions
    stacked = L.Concatenate()([normalized_positions, compressed_features])
    output = get_block(hp.p, hp, stacked, d_out)
    for i in range(hp.p_blocks - 1):
        output = get_block(hp.p, hp, stacked, output)
    trial_name = f'{trial_number}-' + '-'.join(f'{k}.{round(v, 4) if isinstance(v, float) else v}' for k, v in hp.items())
    return K.Model([positions, features], output, name=trial_name), trial_name


@tf.function
def parse_item(example):
    context_features = {'quantum_target': tf.io.FixedLenFeature([], dtype=tf.string),
                        'type': tf.io.FixedLenFeature([], dtype=tf.string),
                        'id': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = {'target_positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'target_features': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'target_masses': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'target_numbers': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'features': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'masses': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'numbers': tf.io.FixedLenSequenceFeature([], dtype=tf.string)}
    context, sequence = tf.io.parse_single_sequence_example(example, context_features=context_features, sequence_features=sequence_features)
    target_positions = tf.reshape(tf.io.parse_tensor(sequence['target_positions'][0], tf.float32), [-1, 3])
    target_features = tf.reshape(tf.io.parse_tensor(sequence['target_features'][0], tf.float32), [-1, 5])
    target_masses = tf.reshape(tf.io.parse_tensor(sequence['target_masses'][0], tf.float32), [-1, 1])
    target_numbers = tf.reshape(tf.io.parse_tensor(sequence['target_numbers'][0], tf.float32), [-1, 1])
    positions = tf.reshape(tf.io.parse_tensor(sequence['positions'][0], tf.float32), [-1, 3])
    features = tf.reshape(tf.io.parse_tensor(sequence['features'][0], tf.float32), [-1, 5])
    numbers = tf.reshape(tf.io.parse_tensor(sequence['numbers'][0], tf.float32), [-1, 1])
    masses = tf.reshape(tf.io.parse_tensor(sequence['masses'][0], tf.float32), [-1, 1])
    numbers = tf.reshape(tf.io.parse_tensor(sequence['numbers'][0], tf.float32), [-1, 1])
    quantum_target = tf.io.parse_tensor(context['quantum_target'], tf.float32)
    features = tf.concat([features, masses, numbers], -1)
    masses = tf.concat([masses, masses, masses], 1)
    n_atoms = tf.shape(positions)[0]
    type = context['type']
    id = context['id']
    return (type, id, n_atoms, target_positions, positions, features, masses, numbers, quantum_target, target_features, target_masses, target_numbers)


def read_shards(datatype):
    print("read_shards", datatype)
    dataset_path = os.path.join('.', 'datasets', 'tfrecord', datatype)
    n_records = len(os.listdir(dataset_path))
    filenames = [os.path.join(dataset_path, str(i) + '.tfrecord') for i in range(n_records)]
    dataset = tf.data.TFRecordDataset(filenames, 'ZLIB')
    dataset = dataset.map(map_func=parse_item, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(1)
    return n_records, dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def move_atom(xyz):
    global atom_index
    xyz = xyz.tolist()
    cmd.translate(xyz, f'id {str(atom_index)}')
    atom_index += 1


def move_atoms(velocities):
    global atom_index
    atom_index = 0
    np.array([move_atom(xyz) for xyz in velocities])


def prepare_pymol():
    cmd.show(GIF_STYLE)
    cmd.unpick()
    util.cbc()


def make_gif(pdb_name, trial_name, pngs_path):
    gif_name = f'{pdb_name}-{trial_name}.gif'
    gif_path = os.path.join(".", "gifs", gif_name)
    imagepaths, images = [], []
    for stackname in os.listdir(pngs_path):
        print('processing', stackname)
        filepath = os.path.join(pngs_path, stackname)
        imagepaths.append(filepath)
    imagepaths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for imagepath in imagepaths:
        image = imageio.imread(imagepath)
        images.append(image)
    print('saving gif to', gif_path)
    imageio.mimsave(gif_path + pdb_name + ".gif", images)


@tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                              tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32)])
def get_loss(initial, xyz):
    return tf.math.square(tf.math.subtract(initial, get_distances(xyz))) * 0.5


@tf.function(input_signature=[tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32)])
def get_distances(xyz):
    xyz = tf.squeeze(xyz, 0)
    d = tf.reshape(tf.reduce_sum(xyz * xyz, 1), [-1, 1])
    return d - 2 * tf.matmul(xyz, xyz, transpose_b=True) + tf.transpose(d)


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
    agent, trial_name = get_agent(trial_number, hp, d_in=D_FEATURES, d_compressed=D_COMPRESSED, d_out=D_OUT)
    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    averages = ema.apply(agent.weights)
    lr = tf.keras.experimental.CosineDecayRestarts(lr, hp.decay)
    optimizer = tf.keras.optimizers.Adam(lr, amsgrad=True)
    writer = tf.summary.create_file_writer(log_dir)

    def run_episode(*args):
        target_features, target_masses, target_numbers = None, None, None
        quantum_target = None
        type = args[0]
        if type is "xyz":
            _, target_positions, positions, features, masses, numbers, quantum_target, gif = args
        elif type is "rxn":
            _, target_positions, positions, features, masses, numbers, target_features, target_masses, target_numbers, gif = args
        else:
            _, target_positions, positions, features, masses, numbers, gif = args
        target_distances = get_distances(target_positions)
        meta = get_loss(target_distances, positions)
        initial_loss = tf.reduce_sum(meta)
        velocities = tf.zeros_like(positions)
        stop = initial_loss * STOP_LOSS_MULTIPLE
        loss = 0.
        for step in range(MAX_STEPS):
            with tf.GradientTape() as tape:
                velocities = velocities + agent([positions, features]) / masses
                positions = tf.math.add(positions, velocities)
                meta = get_loss(target_distances, positions)
            gradients = tape.gradient([meta], agent.trainable_weights)
            optimizer.apply_gradients(zip(gradients, agent.trainable_weights))
            ema.apply(agent.weights)
            loss = tf.reduce_sum(meta)
            new_stop = loss * STOP_LOSS_MULTIPLE
            if tf.math.logical_or(tf.math.greater(loss, stop), tf.math.is_nan(loss)):
                break
            elif tf.math.less(new_stop, stop):
                stop = new_stop
            if gif:
                move_atoms(tf.squeeze(velocities, 0).numpy())
                cmd.zoom()
                cmd.png(os.path.join(pngs_path, f'{str(step)}.png'), width=IMAGE_SIZE, height=IMAGE_SIZE)
        if gif:
            make_gif(pdb_id, trial_name, pngs_path)
        return ((loss - initial_loss) / initial_loss) * 100.

    run_qm9_training_episode = tf.function(run_episode, input_signature=[
        tf.TensorSpec(shape=(), dtype=tf.string),              # type
        tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32),   # target_positions
        tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32),   # positions
        tf.TensorSpec(shape=(1, None, 7), dtype=tf.float32),   # features
        tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32),   # masses
        tf.TensorSpec(shape=(1, None, 1), dtype=tf.float32),   # atomic numbers
        tf.TensorSpec(shape=(15), dtype=tf.float32)])          # quantum_target
    run_rxn_training_episode = tf.function(run_episode, input_signature=[
        tf.TensorSpec(shape=(1), dtype=tf.string),             # type
        tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32),   # target_positions
        tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32),   # positions
        tf.TensorSpec(shape=(1, None, 7), dtype=tf.float32),   # features
        tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32),   # masses
        tf.TensorSpec(shape=(1, None, 1), dtype=tf.float32),   # atomic numbers
        tf.TensorSpec(shape=(1, None, 7), dtype=tf.float32),   # target_features
        tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32),   # target_masses
        tf.TensorSpec(shape=(1, None, 1), dtype=tf.float32)])  # target_numbers
    run_protein_training_episode = tf.function(run_episode, input_signature=[
        tf.TensorSpec(shape=(1), dtype=tf.string),             # type
        tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32),   # target_positions
        tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32),   # positions
        tf.TensorSpec(shape=(1, None, 7), dtype=tf.float32),   # features
        tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32),   # masses
        tf.TensorSpec(shape=(1, None, 1), dtype=tf.float32)])  # atomic numbers

    @tf.function
    def train(qm9, rxn, proteins):
        total_change = 0.
        episode = 0
        change = 0.
        episodes_this_dataset = 0
        for type, id, n_atoms, target_positions, positions, features, masses, numbers, quantum_target, _, _, _ in qm9:
            print("positions shape", positions.shape)
            print("features shape", features.shape)
            print("masses shape", masses.shape)
            print("numbers shape", numbers.shape)
            with tf.device('/gpu:0'):
                change = run_qm9_training_episode(type, target_positions, positions, features, masses, numbers, quantum_target)
                if tf.math.is_nan(change):
                    change = 200.
            tf.print(type, 'episode', episode, 'id', id, 'with', n_atoms, 'atoms', change, "% change (lower is better)")
            tf.summary.scalar('change', change)
            total_change = total_change + change
            episodes_this_dataset = episodes_this_dataset + 1
            episode = episode + 1
            if episode >= EPISODES_PER_TRIAL or episodes_this_dataset >= EPISODES_PER_DATASET:
                break
        episodes_this_dataset = 0
        for type, id, n_atoms, target_positions, positions, features, masses, numbers, _, target_features, target_masses, target_numbers in rxn:
            with tf.device('/gpu:0'):
                change = run_rxn_training_episode(type, target_positions, positions, features, masses, numbers, target_features, target_masses, target_numbers)
            tf.print(type, 'episode', episode, 'id', id, 'with', n_atoms, 'atoms', change, "% change (lower is better)")
            tf.summary.scalar('change', change)
            total_change = total_change + change
            episodes_this_dataset = episodes_this_dataset + 1
            episode = episode + 1
            if episode >= EPISODES_PER_TRIAL or episodes_this_dataset >= EPISODES_PER_DATASET:
                break
        episodes_this_dataset = 0
        for type, id, n_atoms, target_positions, positions, features, masses, numbers, _, _, _, _ in proteins:
            with tf.device('/gpu:0'):
                change = run_protein_training_episode(type, target_positions, positions, features, masses, numbers)
            tf.print(type, 'episode', episode, 'id', id, 'with', n_atoms, 'atoms', change, "% change (lower is better)")
            tf.summary.scalar('change', change)
            total_change = total_change + change
            episodes_this_dataset = episodes_this_dataset + 1
            episode = episode + 1
            if episode >= EPISODES_PER_TRIAL or episodes_this_dataset >= EPISODES_PER_DATASET:
                break
        return total_change

    changes = []
    for repeat_number in range(REPEATS_PER_TRIAL):
        try:
            if repeat_number > 0:
                qm9 = qm9.shuffle(buffer_size=n_qm9_records)
                rxn = rxn.shuffle(buffer_size=n_rxn_records)
                proteins = proteins.shuffle(buffer_size=n_proteins)
            total_change = train(qm9, rxn, proteins)
            total_change = total_change.numpy().item(0)
            print('repeat', repeat_number + 1, 'total change:', total_change, '%')
            changes.append(total_change)
        except Exception as e:
            total_change = EPISODES_PER_TRIAL * 100 * STOP_LOSS_MULTIPLE
            print(e)

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
        averages = [ema.average(weight).numpy() for weight in agent.weights]
        agent.set_weights(averages)
        for movie_number in range(N_MOVIES):
            try:
                pngs_path = os.path.join('.', 'pngs')
                shutil.rmtree(pngs_path)
            except Exception as e:
                print(e)
            os.makedirs(pngs_path)
            pdb_id = random.choice(pedagogy)
            target_positions, positions, features, masses, _, _ = parse_item(load(pdb_id))
            prepare_pymol()
            run_episode(target_positions, positions, features, masses)
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


class PlotCallback(object):
    def __init__(self, path):
        self.path = path

    def __call__(self, results):
        print(results)
        if trial_number > 10:
            _ = plot_evaluations(results)
            plt.savefig(os.path.join(self.path, "evaluations.png"))
            _ = plot_objective(results)
            plt.savefig(os.path.join(self.path, "objective.png"))


def experiment():
    global log_dir, n_qm9_records, qm9, n_rxn_records, rxn, n_proteins, proteins
    log_dir = os.path.join('.', 'runs', str(time.time()))

    if TENSORBOARD:
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir])
        webbrowser.get(using='google-chrome').open(tb.launch()+'#scalars', new=2)


    n_qm9_records, qm9 = read_shards('xyz')
    n_rxn_records, rxn = read_shards('rxn')
    n_proteins, proteins = read_shards('cif')

    checkpoint_path = os.path.join(log_dir, 'checkpoint.pkl')
    checkpointer = skopt.callbacks.CheckpointSaver(checkpoint_path)
    plotter = PlotCallback('.')
    try:
        res = skopt.load(checkpoint_path)
        x0 = res.x_iters
        y0 = res.func_vals
        results = skopt.gp_minimize(trial, dimensions, x0=x0, y0=y0, verbose=True, acq_func=ACQUISITION_FUNCTION, callback=[checkpointer, plotter])
    except Exception as e:
        print(e)
        x0 = hyperpriors
        results = skopt.gp_minimize(trial, dimensions, x0=x0, verbose=True, acq_func=ACQUISITION_FUNCTION, callback=[checkpointer, plotter])
    print(results)


if __name__ == '__main__':
    experiment()

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
proteins = None
trial_number = 0
atom_index = 0
best_trial = ''
best_args = []
best = 12345678900
# parameters
ACQUISITION_FUNCTION = 'EIps'  # 'gp-hedge' if you don't care about speed
STOP_LOSS_MULTIPLE = 1.2
EPISODES_PER_TRIAL = 2
REPEATS_PER_TRIAL = 2
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
    1,  # compressor_blocks,
    'c_conv_wide_deep',  # compressor block type
    1,  # compressor_layers
    512,  # compressor_units
    4,  # pair_blocks
    'p_conv_wide_deep',  # pair block type
    1,  # pair_layers
    512,  # pair_units,
    0.02,  # stddev
    'normal',
    'first',  # norm
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
    input = K.Input((d_features, ))
    if pair:
        input2 = K.Input((d_features, ))
        inputs = L.Concatenate()([input, input2])
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
        block_output = features
        d_output = prior
    else:
        block_output = L.Concatenate(-1)([features, prior])
        d_output = prior.shape[-1]
    d_features = block_output.shape[-1]
    if block_type in ["c_conv_deep", "c_conv_wide_deep"]:
        block_output = ConvKernel(hp, d_features, d_output)(block_output)
    elif block_type in ["p_conv_deep", 'p_conv_wide_deep']:
        block_output = ConvPair(hp, d_features, d_output)(block_output)
    elif block_type in ["c_res_deep", "c_res_wide_deep"]:
        block_output = get_kernel(block_type, hp.c_layers, hp.c_units, hp, block_output.shape[-1], d_output)(block_output)
    elif block_type in ["p_res_deep", "p_res_wide_deep"]:
        block_output = get_kernel(block_type, hp.p_layers, hp.p_units, hp, block_output.shape[-1], d_output)(block_output)
    if hp.norm is 'all':
        block_output = L.BatchNormalization()(block_output)
    if isinstance(prior, int):
        return block_output
    else:
        return L.Add()([prior, block_output])


def get_agent(trial_number, d_in, d_compressed, d_out, hp):
    print('\ntrial', trial_number, '\n')
    [print(f'   {k}={v}') for k, v in hp.items()]
    positions = K.Input((None, 3))
    features = K.Input((None, 13))
    stacked = L.Concatenate(-1)([positions, features])
    normalized = L.BatchNormalization()(stacked) if hp.norm is not 'none' else stacked
    compressed_features = get_block(hp.c, hp, normalized, d_compressed)
    for compressor_block in range(hp.c_blocks - 1):
        compressed_features = get_block(hp.c, hp, normalized, compressed_features)
    output = get_block(hp.p, hp, compressed_features, d_out)
    for i in range(hp.p_blocks - 1):
        output = get_block(hp.p, hp, compressed_features, output)
    trial_name = f'{trial_number}-' + '-'.join(f'{k}.{round(v, 4) if isinstance(v, float) else v}' for k, v in hp.items())
    return K.Model([positions, features], output, name=trial_name), trial_name


@tf.function
def parse_item(example):
    context_features = {'quantum_target': tf.io.FixedLenFeature([], dtype=tf.string),
                        'type': tf.io.FixedLenFeature([], dtype=tf.string),
                        'id': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = {'target_positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'target_features': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'features': tf.io.FixedLenSequenceFeature([], dtype=tf.string)}
    context, sequence = tf.io.parse_single_sequence_example(example, context_features=context_features, sequence_features=sequence_features)
    target_positions = tf.reshape(tf.io.parse_tensor(sequence['target_positions'][0], tf.float32), [-1, 3])
    target_features = tf.reshape(tf.io.parse_tensor(sequence['target_positions'][0], tf.float32), [-1, 13])
    positions = tf.reshape(tf.io.parse_tensor(sequence['positions'][0], tf.float32), [-1, 3])
    features = tf.reshape(tf.io.parse_tensor(sequence['features'][0], tf.float32), [-1, 13])
    quantum_target = tf.io.parse_tensor(sequence['quantum_target'][0], tf.float32)
    elements, masses = features[:, 1], features[:, 0]
    masses = tf.concat([masses, masses, masses], 1)
    return (context['type'], context['id'], target_positions, positions, features, elements, masses,
            tf.shape(positions)[0], target_features, quantum_target)


def read_shards(datatype):
    print("read_shards", datatype)
    dataset_path = os.path.join('.', 'datasets', 'tfrecord', datatype)
    n_records = len(os.listdir(dataset_path))
    filenames = [os.path.join(dataset_path, str(i) + '.tfrecord') for i in range(n_records)]
    dataset = tf.data.TFRecordDataset(filenames, 'ZLIB')
    dataset = dataset.map(map_func=parse_item, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(1)
    return n_data_elements, dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


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
def get_losses(initial, xyz):
    return tf.math.square(tf.math.subtract(initial, get_distances(xyz))) * 0.5


@tf.function(input_signature=[tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32)])
def get_distances(xyz):
    xyz = tf.cast(tf.squeeze(xyz, 0), tf.float16)
    d = tf.reshape(tf.reduce_sum(xyz * xyz, 1), [-1, 1])
    return d - 2 * tf.matmul(xyz, xyz, transpose_b=True) + tf.transpose(d)


@skopt.utils.use_named_args(dimensions=dimensions)
def trial(**kwargs):
    global run_step, best, best_trial, best_args, trial_number, writer, ema, agent, optimizer, hp, proteins
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
    agent, trial_name = get_agent(trial_number, 16, 5, 3, hp)
    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    averages = ema.apply(agent.weights)
    lr = tf.keras.experimental.CosineDecayRestarts(lr, hp.decay)
    optimizer = tf.keras.optimizers.Adam(lr, amsgrad=True)
    writer = tf.summary.create_file_writer(log_dir)

    def run_episode(target_positions, positions, features, masses, gif=False):
        target_distances = get_distances(target_positions)
        meta, overlap = get_losses(target_distances, positions)
        initial_loss = tf.reduce_sum(meta)
        velocities = tf.zeros_like(positions)
        stop = initial_loss * STOP_LOSS_MULTIPLE
        loss = 0.
        for step in tf.range(MAX_STEPS):
            with tf.GradientTape() as tape:
                velocities = velocities + agent([positions, features]) / masses
                positions = tf.math.add(positions, velocities)
                meta = get_losses(target_distances, positions)
            gradients = tape.gradient([meta], agent.trainable_weights)
            optimizer.apply_gradients(zip(gradients, agent.trainable_weights))
            ema.apply(agent.weights)
            total_meta = tf.reduce_sum(meta)
            total_overlap = tf.reduce_sum(overlap)
            loss = total_meta + total_overlap
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

    run_training_episode = tf.function(run_episode, input_signature=[
        tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(1, None, 13), dtype=tf.float32),
        tf.TensorSpec(shape=(1, None, 3), dtype=tf.float32)])

    @tf.function
    def train(qm9, rxns, proteins):
        total_change = 0.
        episode = 0
        change = 0.
        for target_positions, positions, features, masses, pdb_id, n_atoms in proteins:
            with tf.device('/gpu:0'):
                change = run_training_episode(target_positions, positions, features, masses)
            if tf.math.is_nan(change):
                change = 100.
            tf.print('episode', episode, 'pdb id', pdb_id, 'with', n_atoms, 'atoms', change, "% change (lower is better)")
            tf.summary.scalar('change', change)
            total_change = total_change + change
            episode = episode + 1
            if episode >= EPISODES_PER_TRIAL:
                break
        return total_change

    changes = []
    for repeat_number in range(REPEATS_PER_TRIAL):
        try:
            proteins = proteins if repeat_number is 0 else proteins.shuffle(buffer_size=9288)
            total_change = train(proteins)
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
    global log_dir, proteins
    log_dir = os.path.join('.', 'runs', str(time.time()))

    if TENSORBOARD:
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir])
        webbrowser.get(using='google-chrome').open(tb.launch()+'#scalars', new=2)

    datasets = []
    for datatype in ['xyz', 'rxn', 'cif']:
        n_data_elements, dataset = read_shards(datatype)
        datasets.append((datatype, n_data_elements, dataset))
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

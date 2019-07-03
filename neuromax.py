# experiment.py: why? simplify
from plots import plot_objective, plot_evaluations
import matplotlib.pyplot as plt
from tensorboard import program
from attrdict import AttrDict
import tensorflow as tf
from pymol import cmd, util
import multiprocessing
import numpy as np
import webbrowser
import random
import imageio
import shutil
import skopt
import math
import time
import os
from make_dataset import load_pedagogy, load
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.set_cmap("viridis")
B = tf.keras.backend
L = tf.keras.layers
K = tf.keras
# globals
trial_number = 0
atom_index = 0
best_trial = ''
best_args = []
best = 12345678900
# parameters
ACQUISITION_FUNCTION = 'EIps'  # 'gp-hedge' if you don't care about speed
STOP_LOSS_MULTIPLE = 1.2
EPISODES_PER_TRIAL = 64
REPEATS_PER_TRIAL = 6
TENSORBOARD = False
MAX_STEPS = 420
N_RANDOM_STARTS = 10
N_CALLS = 1000
MOVIE_STYLE = "spheres"
IMAGE_SIZE = 256
N_MOVIES = 5
# hyperparameters
dimensions = [
    skopt.space.Integer(1, 6, name='c_blocks'),
    skopt.space.Categorical(['c_res_deep', 'c_res_wide_deep', 'c_conv_deep', 'c_conv_wide_deep'], name="c"),
    skopt.space.Integer(1, 3, name='c_layers'),
    skopt.space.Integer(1, 1024, name='c_units'),
    skopt.space.Integer(1, 6, name='p_blocks'),
    skopt.space.Categorical(['p_res_deep', 'p_res_wide_deep', 'p_conv_deep', 'p_conv_wide_deep'], name='p'),
    skopt.space.Integer(1, 3, name='p_layers'),
    skopt.space.Integer(1, 1024, name='p_units'),
    skopt.space.Real(0.001, 0.1, name='stddev'),
    skopt.space.Categorical(['normal', 'noise', 'drop', 'noisedrop'], name='layer'),
    skopt.space.Categorical(['first', 'all'], name='norm'),
    skopt.space.Real(0.00001, 0.01, name='lr'),
    skopt.space.Integer(10, 1000000, name='decay')]  # major factor in step size
hyperpriors = [
    2,  # compressor_blocks,
    'c_conv_deep',  # compressor block type
    1,  # compressor_layers
    256,  # compressor_units
    2,  # pair_blocks
    'p_conv_deep',  # pair block type
    1,  # pair_layers
    1024,  # pair_units,
    0.02,  # stddev
    'normal',
    'first',  # norm
    0.004,  # lr
    100000]  # decay_steps


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

    @tf.function
    def add_noise(self):
        return self.kernel + tf.random.truncated_normal(tf.shape(self.kernel), stddev=self.stddev), self.bias + tf.random.truncated_normal(tf.shape(self.bias), stddev=self.stddev)

    def call_noise_drop(self, x):
        kernel, bias = self.add_noise()
        kernel = tf.nn.dropout(kernel, 0.5)
        return self.activation(tf.nn.bias_add(B.dot(x, kernel), bias))

    def call_drop(self, x):
        return self.activation(tf.nn.bias_add(B.dot(x, tf.nn.dropout(self.kernel, 0.5)), self.bias))

    def call_noise(self, x):
        kernel, bias = self.add_noise()
        return self.activation(tf.nn.bias_add(B.dot(x, kernel), bias))

    def normal_call(self, x):
        return self.activation(tf.nn.bias_add(B.dot(x, self.kernel), self.bias))


def get_layer(units, hp):
    return NoisyDropConnectDense(units, activation="tanh", layer=hp.layer, stddev=hp.stddev)
    # return L.Dense(units, activation="tanh")


class ConvKernel(L.Layer):
    def __init__(self, hp, d_features, d_output):
        super(ConvKernel, self).__init__()
        self.kernel = get_kernel(hp.c, hp.c_layers, hp.c_units, hp, d_features + d_output, d_output)

    def call(self, inputs):
        return tf.vectorized_map(self.kernel, inputs)


class ConvPair(L.Layer):
    def __init__(self, hp, d_features, d_output):
        super(ConvPair, self).__init__()
        self.kernel = get_kernel(hp.p, hp.p_layers, hp.p_units, hp, (d_features + d_output), d_output, pair=True)

    def call(self, inputs):
        return tf.vectorized_map(lambda a1: tf.reduce_sum(tf.vectorized_map(
                lambda a2: self.kernel([a1, a2]), inputs), axis=0), inputs)


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
    if pair:
        return K.Model([input, input2], output)
    return K.Model(input, output)


def get_block(block_type, hp, features, prior):
    block_output = L.Concatenate(-1)([features, prior])
    d_features = features.shape[-1]
    d_output = prior.shape[-1]
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
    return L.Add()([prior, block_output])


def get_agent(trial_number, d_in, d_compressed, d_out, hp):
    print('\ntrial', trial_number, '\n')
    [print(f'   {k}={v}') for k, v in hp.items()]
    features = K.Input((None, d_in))
    compressed_noise = K.Input((None, d_compressed))
    output_noise = K.Input((None, d_out))
    normalized = L.BatchNormalization()(features) if hp.norm is not 'none' else features
    compressed_features = get_block(hp.c, hp, normalized, compressed_noise)
    for compressor_block in range(hp.c_blocks - 1):
        compressed_features = get_block(hp.c, hp, normalized, compressed_features)
    output = get_block(hp.p, hp, compressed_features, output_noise)
    for i in range(hp.p_blocks - 1):
        output = get_block(hp.p, hp, compressed_features, output)
    trial_name = f'{trial_number}-' + '-'.join(f'{k}.{round(v, 4) if isinstance(v, float) else v}' for k, v in hp.items())
    return K.Model([features, compressed_noise, output_noise], output, name=trial_name), trial_name


@tf.function
def parse_protein(example):
    context_features = {'protein': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = {
        'initial_positions': tf.io.FixedLenSequenceFeature(
            [], dtype=tf.string),
        'positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
        'features': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
        'masses': tf.io.FixedLenSequenceFeature([], dtype=tf.string)
    }
    context, sequence = tf.io.parse_single_sequence_example(example, context_features=context_features, sequence_features=sequence_features)
    initial_positions = tf.reshape(tf.io.parse_tensor(
        sequence['initial_positions'][0], tf.float32), [-1, 3])
    features = tf.reshape(tf.io.parse_tensor(
        sequence['features'][0], tf.float32), [-1, 12])
    masses = tf.reshape(tf.io.parse_tensor(
        sequence['masses'][0], tf.float32), [-1, 1])
    positions = tf.reshape(tf.io.parse_tensor(
        sequence['positions'][0], tf.float32), [-1, 3])
    features = tf.concat([masses, features], 1)
    masses = tf.concat([masses, masses, masses], 1)
    velocities = tf.random.normal(tf.shape(positions))
    forces = tf.zeros(tf.shape(positions))
    return initial_positions, positions, features, masses, forces, velocities, context['protein'], tf.shape(positions)[0]


def read_shards(filenames):
    dataset = tf.data.TFRecordDataset(filenames, 'ZLIB')
    dataset = dataset.shuffle(buffer_size=1237)
    dataset = dataset.map(map_func=parse_protein, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(1)
    return dataset.prefetch(buffer_size=1)


@tf.function
def get_losses(initial, xyz):
    d = get_distances(xyz)
    return tf.losses.mean_squared_error(initial, d), tf.math.exp(-d)


@tf.function
def get_distances(xyz):
    xyz = tf.squeeze(xyz, 0)
    d = tf.reshape(tf.reduce_sum(xyz * xyz, 1), [-1,1])
    return d - 2 * tf.matmul(xyz, xyz, transpose_b=True) + tf.transpose(d)


@skopt.utils.use_named_args(dimensions=dimensions)
def trial(**kwargs):
    global run_step, best, best_trial, best_args, trial_number, writer
    start_time = time.perf_counter()
    hp = AttrDict(kwargs)
    agent, trial_name = get_agent(trial_number, 16, 5, 3, hp)
    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    averages = ema.apply(agent.weights)
    global_step = 0
    optimizer = tf.keras.optimizers.Adam(tf.keras.experimental.CosineDecayRestarts(tf.cast(hp.lr, tf.float32), global_step, hp.decay), amsgrad=True)
    writer = tf.summary.create_file_writer(log_dir)

    # @tf.function
    # def run_step(agent, optimizer, positions, features, masses, forces, velocities):
    #     stacked_features = tf.concat([positions, features], axis=-1)
    #     compressed_noise = tf.random.truncated_normal((1, tf.shape(positions)[1], 5), stddev=hp.stddev)
    #     output_noise = tf.random.truncated_normal(tf.shape(positions), stddev=hp.stddev)
    #     velocities = velocities + (agent([stacked_features, compressed_noise, output_noise]) / masses)
    #     return positions + velocities, velocities

    @tf.function
    def run_episode(agent, optimizer, initial_positions, positions, features, masses, forces, velocities):
        initial_distances = get_distances(initial_positions)
        meta, overlap = get_losses(initial_distances, positions)
        initial_loss = tf.reduce_sum(meta) + tf.reduce_sum(overlap)
        stop = initial_loss * STOP_LOSS_MULTIPLE
        loss = 0.
        for step in tf.range(MAX_STEPS):
            with tf.GradientTape() as tape:
                stacked_features = tf.concat([positions, features], axis=-1)
                compressed_noise = tf.random.truncated_normal((1, tf.shape(positions)[1], 5), stddev=hp.stddev)
                output_noise = tf.random.truncated_normal(tf.shape(positions), stddev=hp.stddev)
                velocities = velocities + (agent([stacked_features, compressed_noise, output_noise]) / masses)
                positions = positions + velocities
                meta, overlap = get_losses(initial_distances, positions)
            gradients = tape.gradient([meta, overlap], agent.trainable_weights)
            optimizer.apply_gradients(zip(gradients, agent.trainable_weights))
            loss = tf.reduce_sum(meta) + tf.reduce_sum(overlap)
            new_stop = loss * STOP_LOSS_MULTIPLE
            if tf.math.logical_or(tf.math.greater(loss, stop), tf.is_nan(loss)):
                break
            elif tf.math.less(new_stop, stop):
                stop = new_stop
        return ((loss - initial_loss) / initial_loss) * 100

    @tf.function
    def train(agent, optimize, proteins):
        total_change = 0.
        episode = 0
        for initial_positions, positions, features, masses, forces, velocities, pdb_id, n_atoms in proteins:
            if episode >= EPISODES_PER_TRIAL:
                break
            change = run_episode(agent, optimizer, initial_positions, positions, features, masses, forces, velocities)
            ema.apply(agent.weights)
            if tf.math.is_nan(change):
                change = 100.
            tf.print('episode', episode, 'pdb id', pdb_id, 'with', n_atoms, 'atoms', tf.math.round(change), "% change (lower is better)")
            tf.contrib.summary.scalar('change', change)
            total_change = total_change + change
            episode = episode + 1
        return total_change

    with writer.as_default():
        changes = []
        for repeat_number in range(REPEATS_PER_TRIAL):
            try:
                total_change = train(agent, optimizer, proteins)
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
    print('median:', median, 'stddev:', stddev)
    print('objective:', objective)
    if math.isnan(objective):
        objective = 100 * EPISODES_PER_TRIAL
    if tf.math.less(objective, best):
        plot_path = os.path.join('.', 'runs', trial_name + '.png')
        K.utils.plot_model(agent, plot_path, show_shapes=True)
        agent.summary()
        averages = [ema.average(weight).numpy() for weight in agent.weights]
        agent.set_weights(averages)
        make_gifs(agent, optimizer, trial_name, hp.stddev, N_MOVIES, IMAGE_SIZE, MAX_STEPS, STOP_LOSS_MULTIPLE)
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
    cmd.set('max_threads', multiprocessing.cpu_count())
    cmd.show(MOVIE_STYLE)
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


def make_gifs(agent, optimizer, trial_name, stddev, n_movies, image_size, max_steps, stop_loss_multiple):
    pngs_path = os.path.join('.', 'pngs')
    pedagogy = load_pedagogy()
    for movie_number in range(n_movies):
        step = 0
        try:
            shutil.rmtree(pngs_path)
        except Exception as e:
            print(e)
        os.makedirs(pngs_path)
        pdb_id = random.choice(pedagogy)
        initial_positions, positions, features, masses, forces, velocities, _, n_atoms = parse_protein(load(pdb_id))
        prepare_pymol()
        [initial_positions, positions, features, masses, forces, velocities] = [tf.expand_dims(x, 0) for x in [initial_positions, positions, features, masses, forces, velocities]]
        velocities = tf.random.truncated_normal(tf.shape(positions), stddev=stddev)
        initial_distances = get_distances(initial_positions)
        stop_loss = stop_loss_multiple * tf.reduce_sum([tf.reduce_sum(l) for l in get_losses(initial_distances, positions)])
        while step < max_steps:
            with tf.GradientTape() as tape:
                stacked_features = tf.concat([positions, features], axis=-1)
                compressed_noise = tf.random.truncated_normal((1, n_atoms, 5), stddev=stddev)
                output_noise = tf.random.truncated_normal(tf.shape(positions), stddev=stddev)
                velocities = velocities + agent([stacked_features, compressed_noise, output_noise]) / masses
                positions = positions + velocities
                losses = get_losses(initial_distances, positions)
            gradients = tape.gradient(losses, agent.trainable_weights)
            optimizer.apply_gradients(zip(gradients, agent.trainable_weights))
            loss = tf.reduce_sum([tf.reduce_sum(l) for l in losses])
            if loss > stop_loss or tf.math.is_nan(loss):
                break
            new_stop = loss * stop_loss_multiple
            if new_stop < stop_loss:
                stop_loss = new_stop
            move_atoms(tf.squeeze(velocities, 0).numpy())
            cmd.zoom("all")
            png_path = os.path.join(pngs_path, f'{str(step)}.png')
            cmd.png(png_path, width=image_size, height=image_size)
            step += 1
        make_gif(pdb_id, trial_name, pngs_path)


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
    log_dir = os.path.join('.', 'runs')

    if TENSORBOARD:
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir])
        webbrowser.get(using='google-chrome').open(tb.launch()+'#scalars', new=2)

    path = os.path.join('.', 'tfrecords')
    filenames = [os.path.join(path, name) for name in os.listdir(path)]
    proteins = read_shards(filenames)
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

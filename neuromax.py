# experiment.py: why? simplify
from plots import plot_objective, plot_evaluations
import matplotlib.pyplot as plt
from tensorboard import program
from attrdict import AttrDict
import numpy as np
import webbrowser
import skopt
import math
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.enable_eager_execution()
plt.set_cmap("viridis")
B = tf.keras.backend
L = tf.keras.layers
K = tf.keras
# globals
trial_number = 0
best_trial = ''
best_args = []
best = 12345678900
# training parameters
PRINT_LOSS_EVERY_STEP = False
ACQUISITION_FUNCTION = 'EIps'  # 'gp-hedge' if you don't care about speed
STOP_LOSS_MULTIPLE = 1.2
EPISODES_PER_TRIAL = 370
REPEATS_PER_TRIAL = 5
TENSORBOARD = False
PLOT_MODEL = True
MAX_STEPS = 420
N_RANDOM_STARTS = 10
N_CALLS = 1000
# hyperparameters
dimensions = [
    skopt.space.Integer(1, 4, name='c_blocks'),
    skopt.space.Categorical(['c_res_deep', 'c_res_wide_deep', 'k_conv_deep', 'k_conv_wide_deep'], name="c"),
    skopt.space.Integer(1, 4, name='c_layers'),
    skopt.space.Integer(1, 512, name='c_units'),
    skopt.space.Integer(1, 4, name='p_blocks'),
    skopt.space.Categorical(['p_res_deep', 'p_res_wide_deep', 'p_conv_deep', 'p_conv_wide_deep'], name='p'),
    skopt.space.Integer(1, 4, name='p_layers'),
    skopt.space.Integer(1, 512, name='p_units'),
    skopt.space.Real(0.001, 0.1, name='stddev'),
    skopt.space.Categorical(['normal', 'noise', 'drop', 'noisedrop'], name='layer'),
    skopt.space.Categorical(['none', 'first', 'all'], name='norm'),
    skopt.space.Real(0.00001, 0.01, name='lr'),
    skopt.space.Integer(10, 1000000, name='decay'),
    skopt.space.Real(0.1, 10, name='kMass')]  # major factor in step size
hyperpriors = [
    1,  # compressor_blocks,
    'c_res_deep',  # compressor block type
    1,  # compressor_layers
    106,  # compressor_units
    1,  # pair_blocks
    'p_conv_deep',  # pair block type
    1,  # pair_layers
    229,  # pair_units,
    0.0389,  # stddev
    'normal',
    'all',  # norm
    0.004,  # lr
    934015,  # decay_steps
    0.3622929]  # kMass


# begin agent
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
        kernel = self.kernel + tf.random.truncated_normal(tf.shape(self.kernel), stddev=self.stddev)
        bias = self.bias + tf.random.truncated_normal(tf.shape(self.bias), stddev=self.stddev)
        return kernel, bias

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
    if block_type in ["k_conv_deep", "k_conv_wide_deep"]:
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
        sequence['features'][0], tf.float32), [-1, 9])
    masses = tf.reshape(tf.io.parse_tensor(
        sequence['masses'][0], tf.float32), [-1, 1])
    positions = tf.reshape(tf.io.parse_tensor(
        sequence['positions'][0], tf.float32), [-1, 3])
    features = tf.concat([masses, features], 1)
    masses = tf.concat([masses, masses, masses], 1)
    velocities = tf.random.normal(tf.shape(positions))
    forces = tf.zeros(tf.shape(positions))
    return initial_positions, positions, features, masses, forces, velocities, context['protein'], tf.shape(positions)[0]


def read_shards():
    path = os.path.join('.', 'tfrecords')
    filenames = [os.path.join(path, name) for name in os.listdir(path)]
    dataset = tf.data.TFRecordDataset(filenames, 'ZLIB')
    dataset = dataset.map(map_func=parse_protein, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset.batch(1)


@tf.function
def get_loss(initial, xyz):
    return tf.losses.mean_pairwise_squared_error(initial, xyz)


@skopt.utils.use_named_args(dimensions=dimensions)
def trial(**kwargs):
    global run_step, best, best_trial, best_args, trial_number, writer
    start_time = time.perf_counter()
    hp = AttrDict(kwargs)

    agent, trial_name = get_agent(trial_number, 13, 5, 3, hp)
    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    averages = ema.apply(agent.weights)

    global_step = tf.train.get_or_create_global_step()
    tf.assign(global_step, 0)
    optimizer = tf.keras.optimizers.Adam(tf.train.cosine_decay_restarts(tf.cast(hp.lr, tf.float32), global_step, hp.decay), amsgrad=True)

    writer = tf.contrib.summary.create_file_writer(log_dir)

    @tf.function
    def run_step(agent, optimizer, positions, features, masses, forces, velocities):
        stacked_features = tf.concat([positions, features], axis=-1)
        compressed_noise = tf.random.truncated_normal((1, tf.shape(positions)[1], 5), stddev=hp.stddev)
        output_noise = tf.random.truncated_normal(tf.shape(positions), stddev=hp.stddev)
        forces = agent([stacked_features, compressed_noise, output_noise])
        velocities = velocities + (forces / masses)
        positions = positions + velocities
        return positions, velocities

    @tf.function
    def run_episode(agent, optimizer, initial_positions, positions, features, masses, forces, velocities):
        masses = masses * hp.kMass
        initial_loss = get_loss(initial_positions, positions)
        stop = initial_loss * 1.04
        if tf.math.equal(stop, 0):
            return 4.
        loss = 0.
        for step in tf.range(MAX_STEPS):
            with tf.GradientTape() as tape:
                positions, velocities = run_step(agent, optimizer, positions, features, masses, forces, velocities)
                loss = get_loss(initial_positions, positions)
            gradients = tape.gradient(loss, agent.trainable_weights)
            optimizer.apply_gradients(zip(gradients, agent.trainable_weights))
            ema.apply(agent.weights)
            loss = tf.reduce_sum(loss)
            if PRINT_LOSS_EVERY_STEP:
                tf.print(step, 'stop', stop, 'loss', loss)
            if tf.math.logical_or(tf.math.greater(loss, stop), tf.math.is_nan(loss)):
                break
            new_stop = loss * STOP_LOSS_MULTIPLE
            if tf.math.less(new_stop, stop):
                stop = new_stop
        return ((loss - initial_loss) / initial_loss) * 100

    @tf.function
    def train(agent, optimize, proteins):
        proteins = proteins.shuffle(buffer_size=1237)
        proteins = proteins.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        total_change = 0.
        episode = 0
        for initial_positions, positions, features, masses, forces, velocities, pdb_id, n_atoms in proteins:
            if episode >= EPISODES_PER_TRIAL:
                break
            change = run_episode(agent, optimizer, initial_positions, positions, features, masses, forces, velocities)
            if tf.math.is_nan(change):
                change = 12345678900.
            tf.print('\n', '^^^ episode', episode, 'pdb id', pdb_id, 'with', n_atoms, 'atoms', tf.math.round(change), "% change (lower is better)\n")
            tf.contrib.summary.scalar('change', change)
            total_change = total_change + change
            episode = episode + 1
        return total_change

    with writer.as_default(), tf.contrib.summary.always_record_summaries():
        changes = []
        for repeat_number in range(REPEATS_PER_TRIAL):
            try:
                total_change = train(agent, optimizer, proteins)
                total_change = total_change.numpy().item(0)
                print('repeat', repeat_number + 1, 'total change:', total_change, '%\n')
                changes.append(total_change)
            except Exception as e:
                total_change = 10000 * STOP_LOSS_MULTIPLE
                print(e)

    median = np.median(changes)
    stddev = np.std(changes)
    loss = median + stddev  # reward skill and consistency
    print('changes:', changes)
    print('median:', median, 'stddev:', stddev)
    print('loss:', loss)
    if math.isnan(loss):
        loss = 1234567890

    if tf.math.less(loss, best):
        if PLOT_MODEL:
            plot_path = os.path.join('.', 'runs', trial_name + '.png')
            K.utils.plot_model(agent, plot_path, show_shapes=True)
            agent.summary()
        averages = [ema.average(weight).numpy() for weight in agent.weights]
        agent.set_weights(averages)
        tf.saved_model.save(agent, os.path.join(log_dir, trial_name + ".h5"))
        best_trial = trial_name
        best = loss
    print('best_trial', best_trial)
    print('best', best)

    del agent, writer
    trial_number += 1

    if ACQUISITION_FUNCTION is 'EIps':
        elapsed = time.perf_counter() - start_time
        print(f'trial {trial_number} done in {elapsed}S')
        return loss, elapsed
    else:
        return loss
# end training


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

    proteins = read_shards()
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

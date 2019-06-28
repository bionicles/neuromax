# experiment.py: why? simplify
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
B = tf.keras.backend
L = tf.keras.layers
K = tf.keras
# globals
trial_number = 0
best_trial = ''
best_args = []
best = 12345678900
# training parameters
ACQUISITION_FUNCTION = 'EIps'  # 'gp-hedge' if you don't care about speed
PRINT_LOSS_EVERY_STEP = True
STOP_LOSS_MULTIPLE = 1.04
EPISODES_PER_TRIAL = 100
REPEATS_PER_TRIAL = 5
MASS_COEFFICIENT = 0.1
TENSORBOARD = False
PLOT_MODEL = True
MAX_STEPS = 420
# hyperparameters
dimensions = [
    skopt.space.Integer(1, 8, name='c_blocks'),
    skopt.space.Categorical(['res', 'k_conv'], name="c"),
    skopt.space.Categorical(['deep', 'wide_deep'], name='c_kernel'),
    skopt.space.Integer(1, 4, name='c_layers'),
    skopt.space.Integer(1, 512, name='c_units'),
    skopt.space.Integer(0, 8, name='p_blocks'),
    skopt.space.Categorical(['deep', 'wide_deep'], name='p_kernel'),
    skopt.space.Integer(1, 4, name='p_layers'),
    skopt.space.Integer(1, 2048, name='p_units'),
    skopt.space.Real(0.001, 0.1, name='noise'),
    skopt.space.Categorical(['dense', 'dropconnect', 'noisy_dropconnect'], name='layer'),
    skopt.space.Categorical(['none', 'first', 'all'], name='norm'),
    skopt.space.Categorical(['linear', 'tanh', 'elu'], name='act'),
    skopt.space.Real(0.00001, 0.01, name='lr'),
    skopt.space.Integer(10, 1000000, name='decay')]
hyperpriors = [
    1,  # compressor_blocks,
    'k_conv',  # compressor
    'wide_deep',  # compressor_kernel
    1,  # compressor_layers
    1,  # compressor_units
    1,  # pair_blocks
    'wide_deep',  # pair_kernel
    1,  # pair_layers
    1,  # pair_kernel_units,
    0.03605158074321749,  # stddev
    'dropconnect',  # dense
    'all',  # batch norm after which layers
    'tanh',  # activation
    0.0026608101595377116,  # lr
    1000]  # decay_steps


# begin agent
class NoisyDropConnectDense(L.Dense):
    def __init__(self, *args, **kwargs):
        self.noise = kwargs.pop('noise')
        super(NoisyDropConnectDense, self).__init__(*args, **kwargs)

    def call(self, x):
        return self.activation(tf.nn.bias_add(B.dot(x, tf.nn.dropout(self.kernel + tf.random.truncated_normal(tf.shape(self.kernel), stddev=self.noise), 0.5)), self.bias + tf.random.truncated_normal(tf.shape(self.bias), stddev=self.noise)))


class DropConnectDense(L.Dense):
    def __init__(self, *args, **kwargs):
        super(DropConnectDense, self).__init__(*args, **kwargs)

    def call(self, x):
        return self.activation(tf.nn.bias_add(B.dot(x, tf.nn.dropout(self.kernel, 0.5)), self.bias))


def get_layer(units, hp):
    if hp.layer is 'dense':
        return L.Dense(units, activation=hp.act)
    elif hp.layer is 'dropconnect':
        return DropConnectDense(units, activation=hp.act)
    else:
        return NoisyDropConnectDense(units, noise=hp.noise, activation=hp.act)


class ConvKernel(L.Layer):
    def __init__(self, hp, d_features, d_output):
        super(ConvKernel, self).__init__()
        self.kernel = get_kernel(hp.c_kernel, hp.c_layers, hp.c_units, hp, d_features + d_output, d_output)

    def call(self, inputs):
        return tf.vectorized_map(self.kernel, inputs)


class ConvPair(L.Layer):
    def __init__(self, hp, d_features, d_output):
        super(ConvPair, self).__init__()
        self.kernel = get_kernel(hp.p_kernel, hp.p_layers, hp.p_units, hp, (d_features + d_output), d_output, pair=True)

    def call(self, inputs):
        return tf.vectorized_map(lambda a1: tf.reduce_sum(tf.vectorized_map(
                lambda a2: self.kernel([a1, a2]), inputs), axis=0), inputs)


def get_kernel(kernel_type, layers, units, hp, d_features, d_output, pair=False):
    input = K.Input((d_features, ))
    if pair:
        input2 = K.Input((d_features, ))
        inputs = L.Concatenate()([input, input2])
        output = get_layer(units, hp)(inputs)
    else:
        output = get_layer(units, hp)(input)
    for layer in range(layers - 1):
        output = get_layer(units, hp)(output)
    if kernel_type is 'wide_deep':
        output = L.Concatenate(-1)([input, output])
    output = get_layer(d_output, hp)(output)
    if pair:
        return K.Model([input, input2], output)
    return K.Model(input, output)


def get_block(block_type, hp, features, prior):
    block_output = L.Concatenate(-1)([features, prior])
    d_features = features.shape[-1]
    d_output = prior.shape[-1]
    if block_type is "k_conv":
        block_output = ConvKernel(hp, d_features, d_output)(block_output)
    elif block_type is "conv_pair":
        block_output = ConvPair(hp, d_features, d_output)(block_output)
    elif block_type is "res":
        block_output = get_kernel(hp.c_kernel, hp.c_layers, hp.c_units, hp, block_output.shape[-1], d_output)(block_output)
    if hp.norm is 'all':
        block_output = L.BatchNormalization()(block_output)
    return L.Add()([prior, block_output])


def get_agent(trial_number, d_in, d_compressed, d_out, hp):
    print('\ntrial', trial_number, '\n')
    [print(f'{k}={v}') for k, v in hp.items()]
    features = K.Input((None, d_in))
    compressed_noise = K.Input((None, d_compressed))
    output_noise = K.Input((None, d_out))
    normalized = L.BatchNormalization()(features) if hp.norm is not 'none' else features
    compressed_features = get_block(hp.c, hp, normalized, compressed_noise)
    for compressor_block in range(hp.c_blocks - 1):
        compressed_features = get_block(hp.c, hp, normalized, compressed_features)
    output = get_block('conv_pair', hp, compressed_features, output_noise)
    for i in range(hp.p_blocks - 1):
        output = get_block('conv_pair', hp, compressed_features, output)
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
    masses = tf.concat([masses, masses, masses], 1) * MASS_COEFFICIENT
    velocities = tf.random.normal(tf.shape(positions))
    forces = tf.zeros(tf.shape(positions))
    return initial_positions, positions, features, masses, forces, velocities


def read_shards():
    path = os.path.join('.', 'tfrecords')
    filenames = [os.path.join(path, name) for name in os.listdir(path)]
    dataset = tf.data.TFRecordDataset(filenames, 'ZLIB')
    dataset = dataset.shuffle(buffer_size=1237)
    dataset = dataset.map(map_func=parse_protein, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(1)
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


@tf.function
def get_losses(initial_positions, positions):
    position_loss = tf.sqrt(K.losses.mean_squared_error(initial_positions, positions))
    shape_loss = tf.sqrt(tf.losses.mean_pairwise_squared_error(initial_positions, positions))
    return position_loss, shape_loss


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
    if PLOT_MODEL:
        plot_path = os.path.join('.', 'runs', trial_name + '.png')
        K.utils.plot_model(agent, plot_path, show_shapes=True)
        agent.summary()

    @tf.function
    def run_step(agent, optimizer, initial_positions, positions, features, masses, forces, velocities):
        stacked_features = tf.concat([positions, features], axis=-1)
        compressed_noise = tf.random.truncated_normal((1, tf.shape(positions)[1], 5), stddev=0.0001)
        output_noise = tf.random.truncated_normal(tf.shape(positions), stddev=0.0001)
        forces = agent([stacked_features, compressed_noise, output_noise])
        velocities = velocities + (forces / masses)
        positions = positions + velocities
        return positions, velocities

    @tf.function
    def run_episode(agent, optimizer, initial_positions, positions, features, masses, forces, velocities):
        initial_position_loss, initial_shape_loss = get_losses(initial_positions, positions)
        initial_loss = tf.reduce_sum(initial_position_loss) + tf.reduce_sum(initial_shape_loss)
        stop = initial_loss * 1.04
        loss = 0.
        for step in tf.range(MAX_STEPS):
            with tf.GradientTape() as tape:
                positions, velocities = run_step(agent, optimizer, initial_positions, positions, features, masses, forces, velocities)
                position_loss, shape_loss = get_losses(initial_positions, positions)
            gradients = tape.gradient([position_loss, shape_loss], agent.trainable_weights)
            optimizer.apply_gradients(zip(gradients, agent.trainable_weights))
            ema.apply(agent.weights)
            loss = tf.reduce_sum(position_loss) + tf.reduce_sum(shape_loss)
            tf.print(step, 'stop', stop, 'loss', loss)
            if tf.math.logical_or(tf.math.greater(loss, stop), tf.math.is_nan(loss)):
                break
            new_stop = loss * STOP_LOSS_MULTIPLE
            if tf.math.less(new_stop, stop):
                stop = new_stop
        return ((loss - initial_loss) / initial_loss) * 100

    @tf.function
    def train(agent, optimize, proteins):
        total_change = 0.
        episode = 0
        for initial_positions, positions, features, masses, forces, velocities in proteins:
            if episode >= EPISODES_PER_TRIAL:
                break
            change = run_episode(agent, optimizer, initial_positions, positions, features, masses, forces, velocities)
            if tf.math.is_nan(change):
                change = 12345678900.
            tf.print('\n', 'protein', episode, tf.math.round(change), "% change (lower is better)\n")
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
    try:
        res = skopt.load(checkpoint_path)
        x0 = res.x_iters
        y0 = res.func_vals
        results = skopt.gp_minimize(trial, dimensions, x0=x0, y0=y0, verbose=True, acq_func=ACQUISITION_FUNCTION, callback=[checkpointer])
    except Exception as e:
        print(e)
        x0 = hyperpriors
        y0 = None
        results = skopt.gp_minimize(trial, dimensions, x0=x0, y0=y0, verbose=True, acq_func=ACQUISITION_FUNCTION, callback=[checkpointer])
    print(results)


if __name__ == '__main__':
    experiment()

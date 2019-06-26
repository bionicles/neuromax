# experiment.py: why? simplify
from tensorboard import program
import webbrowser
import datetime
import skopt
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
best = 10000000000000
# training parameters
STOP_LOSS_MULTIPLE = 1.07
EPISODES_PER_TRIAL = 100
TENSORBOARD = False
PLOT_MODEL = True
MAX_STEPS = 420
# HAND TUNED MODEL PARAMETERS (BAD BION!)
ACTIVATION = 'tanh'
# hyperparameters
d_compressor_kernel_layers = skopt.space.Integer(
                                    1, 4, name='compressor_kernel_layers')
d_compressor_kernel_units = skopt.space.Integer(
                                    1, 512, name='compressor_kernel_units')
d_pair_kernel_layers = skopt.space.Integer(1, 4, name='pair_kernel_layers')
d_pair_kernel_units = skopt.space.Integer(1, 2048, name='pair_kernel_units')
d_blocks = skopt.space.Integer(1, 8, name='blocks')
d_learning_rate = skopt.space.Real(0.00001, 0.01, name='learning_rate')
d_stddev = skopt.space.Real(0.001, 0.1, name='stddev')
d_use_noisy_dropconnect = skopt.space.Categorical([True, False],
                                                  name='use_noisy_dropconnect')
dimensions = [
    d_compressor_kernel_layers,
    d_compressor_kernel_units,
    d_pair_kernel_layers,
    d_pair_kernel_units,
    d_blocks,
    d_learning_rate,
    d_stddev,
    d_use_noisy_dropconnect]
tuned_hyperpriors = [
    2,  # compressor layers
    177,  # compressor units
    2,  # pair kernel layers
    928,  # pair kernel units
    2,  # blocks
    0.0026608101595377116,  # LR
    0.03605158074321749,  # stddev
    True]  # use_noisy_dropconnect
default_hyperpriors = [
    1,  # compressor layers
    1,  # compressor units
    1,  # pair kernel layers
    1,  # pair kernel units
    1,  # blocks
    0.001,  # LR
    0.01,  # stddev
    False]  # use_noisy_dropconnect


# begin model
class NoisyDropConnectDense(L.Dense):
    def __init__(self, *args, **kwargs):
        self.stddev = kwargs.pop('stddev')
        super(NoisyDropConnectDense, self).__init__(*args, **kwargs)

    def call(self, x):
        with tf.device('/GPU:0'):
            return self.activation(tf.nn.bias_add(B.dot(x,
                                                        tf.nn.dropout(
                                                            self.kernel + tf.random.truncated_normal(
                                                                tf.shape(self.kernel),
                                                              stddev=self.stddev), 0.5)),
                self.bias + tf.random.truncated_normal(tf.shape(self.bias),
                                                          stddev=self.stddev)))


def get_layer(units, stddev):
    if USE_NOISY_DROPCONNECT:
        return NoisyDropConnectDense(units, stddev=stddev, activation=ACTIVATION)
    else:
        return L.Dense(units, activation=ACTIVATION)


def get_mlp(features, outputs, layers, units, stddev):
    input = K.Input((features, ))
    output = get_layer(units, stddev)(input)
    # output = L.Dropout(0.5)(output)
    for layer in range(layers - 1):
        output = get_layer(units, stddev)(output)
        # output = L.Dropout(0.5)(output)
    output = get_layer(outputs, stddev)(output)
    return K.Model(input, output)


class ConvKernel(L.Layer):
    def __init__(self,
                 features=13, outputs=5, layers=2, units=128, stddev=0.01):
        super(ConvKernel, self).__init__()
        self.kernel = get_mlp(features, outputs, layers, units, stddev)

    def call(self, inputs):
        return tf.vectorized_map(self.kernel, inputs)


class ConvPair(L.Layer):
    def __init__(self,
                 features=16, outputs=3, layers=2, units=128, stddev=0.01):
        super(ConvPair, self).__init__()
        self.kernel = get_mlp(features, outputs, layers, units, stddev)

    def call(self, inputs):
        return tf.vectorized_map(lambda atom1: tf.reduce_sum(tf.vectorized_map(
            lambda atom2: self.kernel(
                tf.concat([atom1, atom2], 1)), inputs), axis=0), inputs)


def make_block(features, noise_or_output,
               pair_kernel_layers, pair_kernel_units, stddev):
    block_output = L.Concatenate(2)([features, noise_or_output])
    block_output = ConvPair(layers=pair_kernel_layers,
                            units=pair_kernel_units)(block_output)
    return L.Add()([block_output, noise_or_output])


def make_agent(name, d_in, d_out, compressor_kernel_layers,
               compressor_kernel_units, pair_kernel_layers,
               pair_kernel_units, blocks, stddev):
    features = K.Input((None, d_in))
    noise = K.Input((None, d_out))
    normalized = tf.expand_dims(L.BatchNormalization()(tf.squeeze(features, 0)), 0)
    compressed_features = ConvKernel(
        layers=compressor_kernel_layers,
        units=compressor_kernel_units,
        stddev=stddev)(normalized)
    output = make_block(compressed_features, noise,
                        pair_kernel_layers, pair_kernel_units, stddev)
    for i in range(blocks - 1):
        output = make_block(compressed_features, output,
                            pair_kernel_layers, pair_kernel_units, stddev)
    return K.Model([features, noise], output)


def make_linear_baseline(name, d_in, d_out):
    features = K.Input((None, d_in))
    normalized = tf.expand_dims(L.BatchNormalization()(tf.squeeze(features, 0)), 0)
    output = L.Dense(d_out)(normalized)
    return K.Model(features, output, name=name)


def make_deep_baseline(name, d_in, d_out):
    features = K.Input((None, d_in))
    normalized = tf.expand_dims(L.BatchNormalization()(tf.squeeze(features, 0)), 0)
    output = L.Dense(d_out, 'tanh')(normalized)
    output = L.Dense(d_out, 'tanh')(output)
    output = L.Dense(d_out)(output)
    return K.Model(features, output, name=name)


def make_wide_and_deep_baseline(name, d_in, d_out):
    features = K.Input((None, d_in))
    normalized = tf.expand_dims(L.BatchNormalization()(tf.squeeze(features, 0)), 0)
    wide = L.Dense(d_out)(normalized)
    deep = L.Dense(d_in, 'tanh')(normalized)
    deep = L.Dense(d_in, 'tanh')(deep)
    deep = L.Dense(d_out, 'tanh')(deep)
    output = L.Add()([wide, deep])
    return K.Model(features, output, name=name)


def make_baseline(name, d_in, d_out):
    if use_baseline is 'linear':
        return make_linear_baseline('linear', d_in, d_out)
    elif use_baseline is 'deep':
        return make_deep_baseline('deep', d_in, d_out)
    else:
        return make_wide_and_deep_baseline('wide_and_deep', d_in, d_out)
# end model


# begin data
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
    context, sequence = tf.io.parse_single_sequence_example(
        example,
        context_features=context_features, sequence_features=sequence_features)
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
    return initial_positions, positions, features, masses, forces, velocities


def read_shards():
    path = os.path.join('.', 'tfrecords')
    filenames = [os.path.join(path, name) for name in os.listdir(path)]
    dataset = tf.data.TFRecordDataset(filenames, 'ZLIB')
    dataset = dataset.shuffle(buffer_size=1237)
    dataset = dataset.map(map_func=parse_protein,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(1)
    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# end data


# begin training
@tf.function
def compute_loss(initial_positions, positions):
    print('tracing compute_loss')
    return tf.reduce_sum([
        tf.reduce_sum(K.losses.mean_squared_error(initial_positions, positions)),
        tf.losses.mean_pairwise_squared_error(initial_positions, positions)])


@tf.function
def run_episode(agent, optimizer, initial_positions, positions, features, masses, forces, velocities):
    initial_loss = compute_loss(initial_positions, positions)
    stop = initial_loss * 1.04
    loss = 0.
    for step in tf.range(MAX_STEPS):
        positions, velocities, loss = run_step(agent, optimizer, initial_positions, positions, features, masses, forces, velocities)
        if tf.math.greater(loss, stop):
            break
        new_stop = loss * STOP_LOSS_MULTIPLE
        if tf.math.less(new_stop, stop):
            stop = new_stop
    return ((loss - initial_loss) / initial_loss) * 100


@tf.function
def train(agent, optimizer, proteins):
    total_change = 0.
    step = 0
    for initial_positions, positions, features, masses, forces, velocities in proteins:
        if step > EPISODES_PER_TRIAL:
            break
        change = run_episode(agent, optimizer, initial_positions, positions, features, masses, forces, velocities)
        if tf.math.is_nan(change):
            change = 1.618 * STOP_LOSS_MULTIPLE * 1000
        tf.print('\n', 'protein', step, tf.math.round(change), "% change (lower is better)\n")
        tf.contrib.summary.scalar('change', change)
        total_change = total_change + change
        step = step + 1
    return total_change


@skopt.utils.use_named_args(dimensions=dimensions)
def trial(compressor_kernel_layers, compressor_kernel_units,
          pair_kernel_layers, pair_kernel_units, blocks,
          learning_rate, stddev, use_noisy_dropconnect):
    global trial_number, writer, USE_NOISY_DROPCONNECT
    if use_baseline is False:
        trial_name = f'trial:{str(trial_number)}-compressor_kernel_layers:{compressor_kernel_layers}-compressor_kernel_units:{compressor_kernel_units}-pair_kernel_layers:{pair_kernel_layers}-pair_kernel_units:{pair_kernel_units}-blocks:{blocks}-learning_rate:{learning_rate}-stddev:{stddev}-use_noisy_dropconnect:{use_noisy_dropconnect}'
    else:
        trial_name = f'trial:{str(trial_number)}-{use_baseline}'
    trial_log_path = os.path.join(log_dir, trial_name)
    writer = tf.contrib.summary.create_file_writer(trial_log_path)

    global_step = tf.train.get_or_create_global_step()
    tf.assign(global_step, 0)

    learning_rate = tf.cast(learning_rate, tf.float32)
    lr_decayed = tf.train.cosine_decay_restarts(
        learning_rate, global_step, 1000)
    optimizer = tf.keras.optimizers.Adam(lr_decayed, amsgrad=True)

    if use_baseline:
        agent = make_baseline('bob', 13, 3)
    else:
        USE_NOISY_DROPCONNECT = use_noisy_dropconnect
        agent = make_agent('agent', 13, 3,
                           compressor_kernel_layers, compressor_kernel_units,
                           pair_kernel_layers, pair_kernel_units,
                           blocks, stddev)
        print(
            '\n  compressor_kernel_layers', compressor_kernel_layers,
            '\n  compressor_kernel_units', compressor_kernel_units,
            '\n  pair_kernel_layers', pair_kernel_layers,
            '\n  pair_kernel_units', pair_kernel_units,
            '\n  blocks', blocks,
            '\n  learning_rate', learning_rate.numpy().item(0),
            '\n  stddev', stddev,
            '\n  use_noisy_dropconnect', use_noisy_dropconnect, '\n')

    if PLOT_MODEL:
        name = use_baseline if use_baseline else 'agent'
        K.utils.plot_model(agent, name + '.png', show_shapes=True)
        agent.summary()
    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    averages = ema.apply(agent.weights)
    global run_step

    @tf.function
    def run_step(agent, optimizer, initial_positions, positions, features, masses, forces, velocities):
        with tf.GradientTape() as tape:
            forces = agent([tf.concat([positions, features], axis=-1),
                            tf.random.normal(tf.shape(positions))])
            velocities = velocities + (forces / masses)
            positions = positions + velocities
            loss = compute_loss(initial_positions, positions)
        gradients = tape.gradient(loss, agent.trainable_weights)
        optimizer.apply_gradients(zip(gradients, agent.trainable_weights))
        tf.assign_add(tf.train.get_or_create_global_step(), 1)
        averages = ema.apply(agent.weights)
        return positions, velocities, loss

    with writer.as_default(), tf.contrib.summary.always_record_summaries():
        try:
            total_change = train(agent, optimizer, proteins)
            total_change = total_change.numpy().item(0)
            print('total change:', total_change, '%')
        except Exception as e:
            total_change = 100 * STOP_LOSS_MULTIPLE
            print(e)

    global best, best_trial, best_args
    if tf.math.less(total_change, best):
        averages = [ema.average(weight).numpy() for weight in agent.weights]
        agent.set_weights(averages)
        tf.saved_model.save(agent, os.path.join(log_dir, trial_name + ".h5"))
        best_trial = trial_name
        best = total_change
        if use_baseline is False:
            best_args = ['compressor_kernel_layers', compressor_kernel_layers,
                        'compressor_kernel_units', compressor_kernel_units,
                        'pair_kernel_layers', pair_kernel_layers,
                        'pair_kernel_units', pair_kernel_units,
                        'blocks', blocks,
                        'learning_rate', learning_rate.numpy().item(0),
                        'stddev', stddev,
                        'use_noisy_dropconnect', use_noisy_dropconnect]
        else:
            best_args = [use_baseline]
    print('best_trial', best_trial)
    print('best_args', best_args)
    print('best', best)

    del agent, writer
    trial_number += 1
    return total_change
# end training


def main():
    global log_dir, use_baseline, proteins
    log_dir = "runs/" + str(time.time())
    os.makedirs(log_dir)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])
    url = tb.launch()
    webbrowser.get(using='google-chrome').open(url+'#scalars', new=2)

    proteins = read_shards()

    use_baseline = 'linear'
    linear_baseline_results = trial(default_hyperpriors)
    use_baseline = 'deep'
    deep_baseline_results = trial(default_hyperpriors)
    use_baseline = 'wide_and_deep'
    wide_and_deep_baseline_results = trial(default_hyperpriors)

    use_baseline = False

    results = skopt.gp_minimize(trial, dimensions,
                                x0=tuned_hyperpriors, verbose=True)
    print(results)
    print('linear_baseline_results', linear_baseline_results)
    print('deep_baseline_results', deep_baseline_results)
    print('wide_and_deep_baseline_results', wide_and_deep_baseline_results)


if __name__ == '__main__':
    main()

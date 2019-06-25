# experiment.py: why? simplify
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import skopt
import time
import os
import seaborn as sns
sns.set()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.enable_eager_execution()
B = tf.keras.backend
L = tf.keras.layers
K = tf.keras
# training parameters
STOP_LOSS_MULTIPLE = 1.07
PLOT_MODEL = True
MAX_STEPS = 420
# HAND TUNED MODEL PARAMETERS (BAD BION!)
USE_NOISY_DROPCONNECT = False
ACTIVATION = 'tanh'
# hyperparameters
d_compressor_kernel_layers = skopt.space.Integer(
                                    2, 4, name='compressor_kernel_layers')
d_compressor_kernel_units = skopt.space.Integer(
                                    16, 1024, name='compressor_kernel_units')
d_pair_kernel_layers = skopt.space.Integer(2, 4, name='pair_kernel_layers')
d_pair_kernel_units = skopt.space.Integer(16, 2048, name='pair_kernel_units')
d_blocks = skopt.space.Integer(1, 8, name='blocks')
d_learning_rate = skopt.space.Real(0.0001, 0.01, name='learning_rate')
d_decay = skopt.space.Real(0.99, 0.99999, name='decay')
d_stddev = skopt.space.Real(0.001, 0.1, name='stddev')
dimensions = [
    d_compressor_kernel_layers,
    d_compressor_kernel_units,
    d_pair_kernel_layers,
    d_pair_kernel_units,
    d_blocks,
    d_learning_rate,
    d_decay,
    d_stddev]
default_hyperparameters = [
    2,  # compressor layers
    32,  # compressor units
    2,  # pair kernel layers
    257,  # pair kernel units
    2,  # blocks
    0.01,  # LR
    0.999,  # decay
    0.01]  # noise


# begin model
class NoisyDropConnectDense(L.Dense):
    def __init__(self, *args, **kwargs):
        self.stddev = kwargs.pop('stddev')
        super(NoisyDropConnectDense, self).__init__(*args, **kwargs)

    def call(self, x):
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
    compressed_features = ConvKernel(
        layers=compressor_kernel_layers,
        units=compressor_kernel_units,
        stddev=stddev)(features)
    output = make_block(compressed_features, noise,
                        pair_kernel_layers, pair_kernel_units, stddev)
    for i in range(blocks - 1):
        output = make_block(compressed_features, output,
                            pair_kernel_layers, pair_kernel_units, stddev)
    resnet = K.Model([features, noise], output)
    if PLOT_MODEL:
        K.utils.plot_model(resnet, name + '.png', show_shapes=True)
        resnet.summary()
    return resnet
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
def compute_running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def make_plot(changes):
    plt.clf()
    N = len(changes)
    x = [i for i in range(N)]
    running_mean = pd.Series(x).rolling(window=N).mean().iloc[N-1:].values
    plt.plot(running_mean, 'r-')
    plt.scatter(x, changes)
    plt.xlabel('batch')
    plt.ylabel('% change (lower is better)')
    plt.title('neuromax')
    plt.savefig('changes.png')


@tf.function
def compute_loss(initial_positions, positions):
    print('tracing compute_loss')
    return tf.reduce_sum([
        tf.reduce_sum(K.losses.mean_squared_error(initial_positions, positions)),
        tf.losses.mean_pairwise_squared_error(initial_positions, positions)])


@tf.function
def run_step(agent, adam, initial_positions, positions, features, masses, forces, velocities):
    with tf.GradientTape() as tape:
        forces = agent([tf.concat([positions, features], axis=-1),
                        tf.random.normal(tf.shape(positions))])
        velocities = velocities + (forces / masses)
        positions = positions + velocities
        loss = compute_loss(initial_positions, positions)
    gradients = tape.gradient(loss, agent.trainable_weights)
    adam.apply_gradients(zip(gradients, agent.trainable_weights),
                         global_step=tf.train.get_or_create_global_step())
    return positions, velocities, loss


@tf.function
def run_episode(agent, adam, initial_positions, positions, features, masses, forces, velocities):
    initial_loss = compute_loss(initial_positions, positions)
    stop = initial_loss * 1.04
    loss = 0.
    for step in tf.range(MAX_STEPS):
        positions, velocities, loss = run_step(agent, adam, initial_positions, positions, features, masses, forces, velocities)
        tf.print('  ', step,
                 'stop', tf.math.round(stop),
                 'loss', tf.math.round(loss))
        if tf.math.greater(loss, stop):
            break
        new_stop = loss * STOP_LOSS_MULTIPLE
        if tf.math.less(new_stop, stop):
            stop = new_stop
    return ((loss - initial_loss) / initial_loss) * 100


@tf.function
def train(agent, adam, proteins):
    total_change = 0.
    for initial_positions, positions, features, masses, forces, velocities in proteins:
        change = run_episode(agent, adam, initial_positions, positions, features, masses, forces, velocities)
        tf.print('\n', tf.math.round(change), "% change (lower is better)\n")
        total_change = total_change + change
    return total_change


@skopt.utils.use_named_args(dimensions=dimensions)
def trial(compressor_kernel_layers, compressor_kernel_units,
          pair_kernel_layers, pair_kernel_units, blocks,
          learning_rate, decay, stddev):
    print(
        '\n  compressor_kernel_layers', compressor_kernel_layers,
        '\n  compressor_kernel_units', compressor_kernel_units,
        '\n  pair_kernel_layers', pair_kernel_layers,
        '\n  pair_kernel_units', pair_kernel_units,
        '\n  learning_rate', learning_rate, '\n  decay', decay,
        '\n  stddev', stddev, '\n')
    lr_decayed = tf.train.exponential_decay(
        learning_rate, tf.train.get_or_create_global_step(), 1, decay)
    adam = tf.train.AdamOptimizer(lr_decayed)
    agent = make_agent('agent', 13, 3,
                       compressor_kernel_layers, compressor_kernel_units,
                       pair_kernel_layers, pair_kernel_units,
                       blocks, stddev)

    proteins = read_shards()
    total_change = train(agent, adam, proteins)

    global best
    if tf.math.less(total_change, best):
        if not os.path.exists("agents"):
            os.makedirs("agents")
        time_string = str(time.time())
        tf.saved_model.save(agent, "agents/" + time_string + ".h5")
        best = total_change
        best_time_string = time_string
    return best_time_string

    tf.reset_default_graph()
    B.clear_session()
    del agent, adam

    return total_change
# end training


if __name__ == '__main__':
    results = skopt.gp_minimize(trial, dimensions,
                                x0=default_hyperparameters, verbose=True)
    print(results)

# experiment.py: why? simplify
import matplotlib.pyplot as plt
from attrdict import AttrDict
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
best = 0
# training parameters
STOP_LOSS_MULTIPLE = 1.2
BATCHES_PER_TRIAL = 10000
PLOT_MODEL = True
MAX_STEPS = 100
BATCH_SIZE = 2
# HAND TUNED MODEL PARAMETERS (BAD BION!)
USE_NOISY_DROPCONNECT = True
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
    512,  # compressor units
    2,  # pair kernel layers
    1024,  # pair kernel units
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
        return tf.map_fn(lambda atom1: tf.reduce_sum(tf.vectorized_map(
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
    output *= -1
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
    pdb_id = context['protein']
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
    # initial_distances = compute_distances(initial_positions)
    velocities = tf.random.normal(tf.shape(positions))
    forces = tf.zeros(tf.shape(positions))
    num_atoms = tf.cast(tf.shape(positions)[0], tf.float32)
    num_atoms_squared = tf.math.square(num_atoms)
    return (initial_positions, positions, features, masses,
            forces, velocities, num_atoms, num_atoms_squared, pdb_id)


def attrdict_for(p):
    return AttrDict({
        'initial_positions': p[0],
        'positions': p[1],
        'features': p[2],
        'masses': p[3],
        # 'initial_distances': p[4],
        'forces': p[4],
        'velocities': p[5],
        'num_atoms': p[6],
        'num_atoms_squared': p[7],
        'pdb_id': p[8]
    })


def read_shards():
    path = os.path.join('.', 'tfrecords')
    recordpaths = [os.path.join(path, name) for name in os.listdir(path)]
    dataset = tf.data.TFRecordDataset(recordpaths, 'ZLIB')
    dataset = dataset.shuffle(buffer_size=len(recordpaths))
    dataset = dataset.map(map_func=parse_protein)
    dataset = dataset.batch(1)
    return dataset.prefetch(buffer_size=BATCH_SIZE)
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


# @tf.function
# def compute_distances(positions):
#     positions = tf.squeeze(positions)
#     distances = tf.reshape(tf.reduce_sum(positions * positions, 1), [-1, 1])
#     return distances - 2 * tf.matmul(
#         positions, tf.transpose(positions)) + tf.transpose(distances)


@tf.function
def compute_loss(p):
    print('tracing compute_loss')
    position_error = K.losses.mean_squared_error(
        p.initial_positions, p.positions) / p.num_atoms
    # shape_error = K.losses.mean_squared_error(
    #     p.initial_distances, compute_distances(p.positions)) / p.num_atoms_squared
    shape_error = tf.losses.mean_pairwise_squared_error(p.initial_positions, p.positions)
    return position_error + shape_error


@tf.function
def compute_batch_mean_loss(batch_losses):
    print('tracing compute_batch_mean_loss')
    return tf.reduce_mean(tf.convert_to_tensor(
        [tf.reduce_sum(loss) for loss in batch_losses]))


def step_agent_on_protein(agent, p):
    print('tracing step_agent_on_protein')
    atoms = tf.concat([p.positions, p.features], axis=-1)
    noise = tf.random.normal(p.positions.shape)
    forces = agent([atoms, noise])
    new_velocities = p.velocities + (forces / p.masses)
    new_positions = p.positions + new_velocities
    return AttrDict({
        'initial_positions': p.initial_positions,
        # 'initial_distances': p.initial_distances,
        'features': p.features,
        'positions': new_positions,
        'velocities': new_velocities,
        'num_atoms_squared': p.num_atoms_squared,
        'num_atoms': p.num_atoms,
        'forces': forces,
        'masses': p.masses,
        'pdb_id': p.pdb_id
    })


def run_episode(adam, agent, batch, invoke):
    print('tracing run_episode')
    initial_batch_losses = [compute_loss(p) for p in batch]
    initial_batch_mean_loss = compute_batch_mean_loss(initial_batch_losses)
    stop = initial_batch_mean_loss * 1.04
    batch_mean_loss = tf.cast(0, tf.float32)
    for step in tf.range(MAX_STEPS):
        with tf.GradientTape() as tape:
            batch = [invoke(agent, p) for p in batch]
            batch_losses = [compute_loss(p) for p in batch]
        gradients = tape.gradient(batch_losses, agent.trainable_weights)
        adam.apply_gradients(zip(gradients, agent.trainable_weights),
                             global_step=tf.train.get_or_create_global_step())
        batch_mean_loss = compute_batch_mean_loss(batch_losses)
        tf.print('  ', step,
                 'stop', tf.math.round(stop),
                 'loss', tf.math.round(batch_mean_loss))
        if tf.math.greater(batch_mean_loss, stop):
            break
        new_stop = batch_mean_loss * STOP_LOSS_MULTIPLE
        if tf.math.less(new_stop, stop):
            stop = new_stop
    change = (batch_mean_loss - initial_batch_mean_loss)
    change /= initial_batch_mean_loss
    change *= 100
    tf.print('\n', tf.math.round(change), "% change (lower is better)")
    return change


@skopt.utils.use_named_args(dimensions=dimensions)
def trial(compressor_kernel_layers, compressor_kernel_units,
          pair_kernel_layers, pair_kernel_units, blocks,
          learning_rate, decay, stddev):
    lr_decayed = tf.train.exponential_decay(
        learning_rate, tf.train.get_or_create_global_step(), 1, decay)
    adam = tf.train.AdamOptimizer(lr_decayed)
    agent = make_agent('agent', 13, 3,
                       compressor_kernel_layers, compressor_kernel_units,
                       pair_kernel_layers, pair_kernel_units,
                       blocks, stddev)
    invoke = tf.function(step_agent_on_protein)
    current_agent_episode_graph = tf.function(run_episode)
    total_change, batch_number, batch = 0, 0, []
    proteins = read_shards()
    for protein in proteins:
        if batch_number is BATCHES_PER_TRIAL:
            break
        if len(batch) < BATCH_SIZE:
            batch.append(attrdict_for(protein))
        else:
            print('\n  batch', batch_number, 'LR', adam._lr().numpy())
            [tf.print(p.pdb_id, p.num_atoms) for p in batch]
            print(
                '\n  compressor_kernel_layers', compressor_kernel_layers,
                '\n  compressor_kernel_units', compressor_kernel_units,
                '\n  pair_kernel_layers', pair_kernel_layers,
                '\n  pair_kernel_units', pair_kernel_units,
                '\n  learning_rate', learning_rate, '\n  decay', decay,
                '\n  stddev', stddev, '\n')
            try:
                loss_change = current_agent_episode_graph(adam, agent, batch, invoke)
                total_change = total_change + loss_change.numpy().item()
                # changes.append(loss_change.numpy().item(0))
                # # make_plot(changes)
            except Exception as e:
                total_change += 10
                print(e)
            batch_number += 1
            batch = []

    global best
    if total_change < best:
        if not os.path.exists("agents"):
            os.makedirs("agents")
        tf.saved_model.save(agent, "agents/" + str(time.time()) + ".h5")
        best = total_change

    tf.reset_default_graph()
    B.clear_session()
    del agent, adam

    return total_change
# end training


if __name__ == '__main__':
    results = skopt.gp_minimize(trial, dimensions,
                                x0=default_hyperparameters, verbose=True)
    print(results)

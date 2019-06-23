# experiment.py: why? simplify
import matplotlib.pyplot as plt
from attrdict import AttrDict
import pandas as pd
import numpy as np
import skopt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.enable_eager_execution()
B = tf.keras.backend
L = tf.keras.layers
K = tf.keras
best = 0
# training parameters
STOP_LOSS_MULTIPLE = 1.04
PLOT_MODEL = True
MAX_STEPS = 100
BATCH_SIZE = 4
# hyperparameters
d_compressor_kernel_layers = skopt.space.Integer(
                                    1, 4, name='compressor_kernel_layers')
d_compressor_kernel_units = skopt.space.Integer(
                                    16, 1024, name='compressor_kernel_units')
d_pair_kernel_layers = skopt.space.Integer(1, 4, name='pair_kernel_layers')
d_pair_kernel_units = skopt.space.Integer(1, 2048, name='pair_kernel_units')
d_blocks = skopt.space.Integer(1, 4, name='blocks')
d_block_layers = skopt.space.Integer(1, 4, name='block_layers')
d_learning_rate = skopt.space.Real(0.0001, 0.01, name='learning_rate')
d_decay = skopt.space.Real(0.00001, 0.01, name='decay')
d_stddev = skopt.space.Real(0.00001, 1, name='stddev')
dimensions = [
    d_compressor_kernel_layers,
    d_compressor_kernel_units,
    d_pair_kernel_layers,
    d_pair_kernel_units,
    d_blocks,
    d_block_layers,
    d_learning_rate,
    d_decay,
    d_stddev]
default_hyperparameters = [
    2,  # compressor layers
    1024,  # compressor units
    2,  # pair kernel layers
    1024,  # pair kernel units
    2,  # blocks
    2,  # block layers
    0.001,  # LR
    0.0001,  # decay
    0.01]  # noise


# begin model
class NoisyDropConnectDense(L.Dense):
    def __init__(self, *args, **kwargs):
        self.stddev = kwargs.pop('stddev')
        super(NoisyDropConnectDense, self).__init__(*args, **kwargs)

    def call(self, x):
        kernel = self.kernel + tf.random.truncated_normal(self.kernel.shape,
                                                          stddev=self.stddev)
        kernel = B.dropout(kernel, 0.5)
        bias = self.bias + tf.random.truncated_normal(self.bias.shape,
                                                      stddev=self.stddev)
        return self.activation(tf.nn.bias_add(B.dot(x, kernel), bias))


def get_layer(units, stddev):
    return NoisyDropConnectDense(units, stddev=stddev, activation='tanh')


def get_mlp(features, outputs, layers, units, stddev):
    input = K.Input((features, ))
    output = get_layer(units, stddev)(input)
    for layer in range(layers - 1):
        output = get_layer(units, stddev)(output)
    output = get_layer(outputs, stddev)(output)
    return K.Model(input, output)


class ConvKernel(L.Layer):
    def __init__(self,
                 features=16, outputs=5, layers=2, units=128, stddev=0.01):
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
        return tf.map_fn(lambda atom1: tf.reduce_sum(tf.map_fn(
            lambda atom2: self.kernel(
                tf.concat([atom1, atom2], 1)), inputs), axis=0), inputs)


def make_block(features, noise_or_output, block_layers,
               pair_kernel_layers, pair_kernel_units, stddev):
    block_output = L.Concatenate(2)([features, noise_or_output])
    for layer_n in range(block_layers - 1):
        block_output = ConvPair(layers=pair_kernel_layers,
                                units=pair_kernel_units)(block_output)
        block_output = L.Concatenate(2)([features, block_output])
    block_output = ConvPair(layers=pair_kernel_layers,
                            units=pair_kernel_units)(block_output)
    return L.Add()([block_output, noise_or_output])


def make_agent(name, d_in, d_out, compressor_kernel_layers,
               compressor_kernel_units, pair_kernel_layers,
               pair_kernel_units, blocks, block_layers, stddev):
    features = K.Input((None, d_in))
    noise = K.Input((None, d_out))
    compressed_features = ConvKernel(
        layers=compressor_kernel_layers,
        units=compressor_kernel_units,
        stddev=stddev)(features)
    output = make_block(compressed_features, noise, block_layers,
                        pair_kernel_layers, pair_kernel_units, stddev)
    for i in range(blocks - 1):
        output = make_block(compressed_features, output, block_layers,
                            pair_kernel_layers, pair_kernel_units, stddev)
    output *= -1
    resnet = K.Model([features, noise], output)
    if PLOT_MODEL:
        K.utils.plot_model(resnet, name + '.png', show_shapes=True)
        resnet.summary()
    return resnet
# end model


# begin data
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
    initial_distances = compute_distances(initial_positions)
    return initial_positions, positions, features, masses, initial_distances


def read_shards():
    path = os.path.join('.', 'tfrecords')
    recordpaths = [os.path.join(path, name) for name in os.listdir(path)]
    dataset = tf.data.TFRecordDataset(recordpaths, 'ZLIB')
    dataset = dataset.shuffle(42)
    dataset = dataset.map(map_func=parse_protein)
    dataset = dataset.batch(1)
    return dataset.prefetch(buffer_size=BATCH_SIZE)
# end data


# begin training
@tf.function
def compute_distances(positions):
    positions = tf.squeeze(positions)
    distances = tf.reduce_sum(positions * positions, 1)
    distances = tf.reshape(distances, [-1, 1])
    return distances - 2 * tf.matmul(
        positions, tf.transpose(positions)) + tf.transpose(distances)


@tf.function
def compute_loss(p):
    position_error = K.losses.mean_squared_error(
        p.initial_positions, p.positions) / p.num_atoms
    distances = compute_distances(p.positions)
    shape_error = K.losses.mean_squared_error(
        p.initial_distances, distances)
    shape_error /= p.num_atoms_squared
    return position_error + shape_error


def compute_batch_mean_loss(batch_losses):
    batch_mean_loss_tensor = tf.reduce_mean(tf.convert_to_tensor(
        [tf.reduce_sum(loss) for loss in batch_losses]))
    return batch_mean_loss_tensor.numpy()


def step_agent_on_protein(agent, p):
    atoms = tf.concat([p.positions, p.velocities, p.features], axis=-1)
    noise = tf.random.normal(p.positions.shape)
    forces = agent([atoms, noise])
    p.velocities += (forces / p.masses)
    p.positions += p.velocities
    return AttrDict({
        'initial_positions': p.initial_positions,
        'initial_distances': p.initial_distances,
        'features': p.features,
        'positions': p.positions,
        'velocities': p.velocities,
        'num_atoms_squared': p.num_atoms_squared,
        'num_atoms': p.num_atoms,
        'forces': forces,
        'masses': p.masses
    })


def attrdict_for(p):
    # initial_positions, positions, features, masses, initial_distances
    num_atoms = tf.dtypes.cast(p[0].shape[1], dtype=tf.float32)
    return AttrDict({
        'num_atoms': num_atoms,
        'num_atoms_squared': tf.square(num_atoms),
        'initial_distances': p[4],
        'velocities': tf.random.normal(p[1].shape),
        'forces': tf.zeros_like(p[0]),
        'initial_positions': p[0],
        'positions': p[1],
        'features': p[2],
        'masses': p[3],
    })


def run_episode(adam, agent, batch):
    initial_batch_losses = [compute_loss(p) for p in batch]
    initial_batch_mean_loss = compute_batch_mean_loss(initial_batch_losses)
    stop = initial_batch_mean_loss * 1.04
    episode_loss = 0
    for step in range(MAX_STEPS):
        with tf.GradientTape() as tape:
            batch = [step_agent_on_protein(agent, p) for p in batch]
            batch_losses = [compute_loss(p) for p in batch]
            gradients = tape.gradient(batch_losses, agent.trainable_weights)
            adam.apply_gradients(zip(gradients, agent.trainable_weights))
        batch_mean_loss = compute_batch_mean_loss(batch_losses)
        print(f'  {step:02d} stop {int(stop)} loss {int(batch_mean_loss)}')
        episode_loss += batch_mean_loss
        if batch_mean_loss > stop:
            break
        if batch_mean_loss * STOP_LOSS_MULTIPLE < stop:
            stop = batch_mean_loss * STOP_LOSS_MULTIPLE
    change = (batch_mean_loss - initial_batch_mean_loss)
    change /= initial_batch_mean_loss
    change *= 100
    print('\n', round(change, 1), "% change (lower is better)")
    return agent, change


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


@skopt.utils.use_named_args(dimensions=dimensions)
def trial(compressor_kernel_layers, compressor_kernel_units,
          pair_kernel_layers, pair_kernel_units, blocks, block_layers,
          learning_rate, decay, stddev):
    lr_decayed = tf.train.exponential_decay(
        learning_rate, tf.train.get_or_create_global_step(), 1000, decay)
    adam = K.optimizers.Adam(lr_decayed)
    agent = make_agent('agent', 16, 3,
                       compressor_kernel_layers, compressor_kernel_units,
                       pair_kernel_layers, pair_kernel_units,
                       blocks, block_layers, stddev)

    batch_number, trial_loss, batch, changes = 0, 0, [], []
    proteins = read_shards()
    for protein in proteins:
        if len(batch) < BATCH_SIZE:
            try:
                batch.append(attrdict_for(protein))
            except Exception as e:
                print(e)
        else:
            print('\n  batch', batch_number)
            try:
                agent, loss_change = run_episode(adam, agent, batch)
                trial_loss += loss_change
                changes.append(loss_change.item(0))
                make_plot(changes)
            except Exception as e:
                print(e)
            batch_number += 1
            batch = []

    global best
    if trial_loss < best:
        save_path = os.path.join('.', 'agents', str(time.time()) + '.h5')
        tf.saved_model.save(agent, save_path)
        best = trial_loss

    B.clear_session()
    del agent

    return trial_loss
# end training


if __name__ == '__main__':
    skopt.gp_minimize(trial, dimensions, x0=default_hyperparameters)

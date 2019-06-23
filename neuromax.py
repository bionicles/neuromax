# experiment.py: why? simplify
from attrdict import AttrDict
import tensorflow as tf
import random
import skopt
import time
import os
tf.compat.v1.enable_eager_execution()
B = tf.keras.backend
L = tf.keras.layers
K = tf.keras
best = 0
# training parameters
STOP_LOSS_MULTIPLE = 1.04
PLOT_MODEL = False
MAX_STEPS = 100
BATCH_SIZE = 2
# hyperparameters
d_compressor_kernel_units = skopt.space.Integer(
                                    16, 2048, name='compressor_kernel_units')
d_compressor_kernel_layers = skopt.space.Integer(
                                    1, 9, name='compressor_kernel_layers')
d_pair_kernel_units = skopt.space.Integer(16, 2048, name='pair_kernel_units')
d_pair_kernel_layers = skopt.space.Integer(1, 9, name='pair_kernel_layers')
d_blocks = skopt.space.Integer(1, 9, name='blocks')
d_block_layers = skopt.space.Integer(1, 9, name='block_layers')
d_learning_rate = skopt.space.Real(0.0001, 0.01, name='learning_rate')
d_decay = skopt.space.Real(0.0001, 0.01, name='decay')
dimensions = [
    d_compressor_kernel_layers,
    d_compressor_kernel_units,
    d_pair_kernel_layers,
    d_pair_kernel_units,
    d_blocks,
    d_block_layers,
    d_learning_rate,
    d_decay]
default_hyperparameters = [
    2,
    128,
    2,
    128,
    2,
    2,
    0.01,
    0.001]


# begin model
# class DropConnectDense(L.Dense):
#     def call(self, inputs):
#         if random.random() > 0.5:
#             kernel = B.dropout(self.kernel, 0.5)
#         else:
#             kernel = self.kernel
#         outputs = B.dot(inputs, kernel)
#         return self.activation(outputs)


def get_layer(units):
    return L.Dense(units, activation='tanh', use_bias=False)


def get_mlp(features, outputs, layers, units):
    input = K.Input((features, ))
    output = get_layer(units)(input)
    for layer in range(layers):
        output = get_layer(units)(output)
    output = get_layer(outputs)(output)
    return K.Model(input, output)


class ConvKernel(L.Layer):
    def __init__(self, features=16, outputs=5, layers=2, units=128):
        super(ConvKernel, self).__init__()
        self.kernel = get_mlp(features, outputs, layers, units)

    def call(self, inputs):
        return tf.vectorized_map(self.kernel, inputs)


class ConvPair(L.Layer):
    def __init__(self, features=16, outputs=3, layers=2, units=128):
        super(ConvPair, self).__init__()
        self.kernel = get_mlp(features, outputs, layers, units)

    def call(self, inputs):
        return tf.map_fn(lambda atom1: tf.reduce_sum(tf.map_fn(
            lambda atom2: self.kernel(
                tf.concat([atom1, atom2], 1)), inputs), axis=0), inputs)


def make_block(
        features, noise_or_output, block_layers,
        pair_kernel_layers, pair_kernel_units):
    block_output = L.Concatenate(2)([features, noise_or_output])
    for layer_n in range(block_layers):
        block_output = ConvPair(layers=pair_kernel_layers,
                                units=pair_kernel_units)(block_output)
        block_output = L.Concatenate(2)([features, block_output])
    block_output = ConvPair(layers=pair_kernel_layers,
                            units=pair_kernel_units)(block_output)
    return L.Add()([block_output, noise_or_output])


def make_agent(name, d_in, d_out, compressor_kernel_layers,
               compressor_kernel_units, pair_kernel_layers,
               pair_kernel_units, blocks, block_layers):
    features = K.Input((None, d_in))
    noise = K.Input((None, d_out))
    compressed_features = ConvKernel(
        layers=compressor_kernel_layers,
        units=compressor_kernel_units)(features)
    output = make_block(compressed_features, noise, block_layers,
                        pair_kernel_layers, pair_kernel_units)
    for i in range(1, blocks):
        output = make_block(compressed_features, output, block_layers,
                            pair_kernel_layers, pair_kernel_units)
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
    initial_distances = compute_distances(positions)
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
def step_agent_on_protein(agent, p):
    noise = tf.random.normal(p.positions.shape)
    forces = agent([tf.concat([p.positions, p.velocities, p.features], axis = -1), noise])
    p.velocities += forces / p.masses
    p.positions += p.velocities
    return AttrDict({
        'initial_positions': p[0],
        'initial_distances': compute_distances(p[0]),
        'features': p.features,
        'positions': p.positions,
        'velocities': p.velocities,
        'num_atoms': p.num_atoms,
        'forces': forces,
        'masses': p.masses
    })


@tf.function
def compute_distances(positions):
    positions = tf.squeeze(positions)
    distances = tf.reduce_sum(positions * positions, 1)
    distances = tf.reshape(distances, [-1, 1])
    return distances - 2 * tf.matmul(
        positions, tf.transpose(positions)) + tf.transpose(distances)


def loss(p):
    position_error = K.losses.mean_squared_error(
        p.initial_positions, p.positions) / p.num_atoms
    shape_error = K.losses.mean_squared_error(
        p.initial_distances, compute_distances(p.positions)) / p.num_atoms
    kinetic_energy = (p.masses / 2) * (p.velocities ** 2)
    potential_energy = p.forces * -1
    action = kinetic_energy - potential_energy
    action = tf.reduce_sum(action, axis = -1)
    return tf.reduce_sum([position_error, shape_error, action])


def run_episode(adam, agent, batch):
    trailing_stop_loss = STOP_LOSS_MULTIPLE * tf.reduce_sum([
        loss(p) for p in batch])
    episode_loss = False, 0
    with tf.GradientTape() as tape:
        for step in range(MAX_STEPS):
            batch = [step_agent_on_protein(agent, p) for p in batch]
            # only do loss/train stuff sometimes to run faster!
            if random.random() > 0.5:
                batch_error = tf.reduce_sum([loss(p) for p in batch])
                gradients = tape.gradient(episode_loss, agent.trainable_weights)
                adam.apply_gradients(zip(gradients, agent.trainable_weights))
                episode_loss += batch_error
                if batch_error * STOP_LOSS_MULTIPLE < trailing_stop_loss:
                    trailing_stop_loss = batch_error * STOP_LOSS_MULTIPLE
                elif batch_error > trailing_stop_loss:
                    break

    return episode_loss


def attrdict_for(p):
    # initial_positions, positions, features, masses, initial_distances
    return AttrDict({
        'num_atoms': tf.dtypes.cast(p[0].shape[1], dtype=tf.float32),
        'initial_distances': p[4],
        'velocities': tf.random.normal(p[1].shape),
        'forces': tf.zeros_like(p[0]),
        'initial_positions': p[0],
        'positions': p[1],
        'features': p[2],
        'masses': p[3],
    })


@skopt.utils.use_named_args(dimensions=dimensions)
def trial(compressor_kernel_layers, compressor_kernel_units,
          pair_kernel_layers, pair_kernel_units, blocks, block_layers,
          learning_rate, decay):
    lr_decayed = tf.train.exponential_decay(
        learning_rate, tf.train.get_or_create_global_step(), 1000, decay)
    adam = K.optimizers.Adam(lr_decayed)
    agent = make_agent('agent', 16, 3,
                       compressor_kernel_layers, compressor_kernel_units,
                       pair_kernel_layers, pair_kernel_units,
                       blocks, block_layers)

    batch_number, trial_loss, batch = 0, 0, []
    proteins = read_shards()
    for protein in proteins:
        if len(batch) < BATCH_SIZE:
            batch.append(attrdict_for(protein))
        else:
            print('batch', batch_number)
            agent, episode_loss = run_episode(adam, agent, batch)
            trial_loss += episode_loss
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
    skopt.gp_minimize(trial, dimensions)

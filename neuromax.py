# neuromax.py - why?: 1 simple file with functions over classes
import tensorflow.keras.layers as L
import tensorflow.keras as K
import tensorflow as tf
import skopt
import time
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.enable_eager_execution()
# globals
best_cumulative_improvement = 0
experiment = 0
# simulation
IMAGE_SIZE = 256
POSITION_VELOCITY_LOSS_WEIGHT, SHAPE_LOSS_WEIGHT = 100, 1
MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE = 8, 64
MIN_STEPS_IN_UNDOCK, MAX_STEPS_IN_UNDOCK = 1, 1
MIN_STEPS_IN_UNFOLD, MAX_STEPS_IN_UNFOLD = 0, 1
SCREENSHOT_EVERY = 10
NOISE = 0.01
WARMUP = 1000
BUFFER = 42
# dataset
PEDAGOGY_FILE_NAME = 'less-than-164kd-9-chains.csv'
BIGGEST_FIRST_IF_NEG = 1
RANDOM_PROTEINS = True
N_PARALLEL_CALLS = 1
BATCH_SIZE = 1
ITERATOR_BATCH = 2
# training
N_RANDOM_TRIALS, N_TRIALS = 1, 1
STOP_LOSS_MULTIPLIER = 1.04
POSITION_ERROR_WEIGHT = 1
ACTION_ERROR_WEIGHT = 1
SHAPE_ERROR_WEIGHT = 1
N_EPISODES = 10000
N_STEPS = 100
VERBOSE = True
# hyperparameters
d_blocks = skopt.space.Integer(1, 9, name='blocks')
d_block_layers = skopt.space.Integer(1, 9, name='block_layers')
d_compressor_kernel_units = skopt.space.Integer(16, 1024, name='compressor_kernel_units')
d_compressor_kernel_layers = skopt.space.Integer(1, 9, name='compressor_kernel_layers')
d_pair_kernel_units = skopt.space.Integer(16, 1024, name='pair_kernel_units')
d_pair_kernel_layers = skopt.space.Integer(1, 9, name='pair_kernel_layers')
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
                d_decay
            ]

default_hyperparameters = [
    2,
    512,
    2,
    512,
    2,
    2,
    0.001,
    0.0001
]


# begin model
def get_layer(units):
    return L.Dense(units, activation='tanh')


def get_mlp(features, outputs, layers, units):
    input = K.Input((features, ))
    output = get_layer(units)(input)
    for layer in range(layers):
        output = get_layer(units)(output)
    output = get_layer(outputs)(output)
    return K.Model(input, output)


class ConvKernel(L.Layer):
    def __init__(self, features=16, outputs=5, layers=2, units=512):
        super(ConvKernel, self).__init__()
        self.kernel = get_mlp(features, outputs, layers, units)

    def call(self, inputs):
        return tf.vectorized_map(self.kernel, inputs)


class ConvPair(L.Layer):
    def __init__(self, features=16, outputs=3, layers=2, units=512):
        super(ConvPair, self).__init__()
        self.kernel = get_mlp(features, outputs, layers, units)

    def call(self, inputs):
        return tf.map_fn(lambda atom1: tf.reduce_sum(tf.map_fn(lambda atom2: self.kernel(tf.concat([atom1, atom2], 1)), inputs), axis=0), inputs)


def make_block(features, noise_or_output, block_layers, pair_kernel_layers, pair_kernel_units):
    block_output = L.Concatenate(2)([features, noise_or_output])
    for layer_n in range(block_layers):
        block_output = ConvPair(layers=pair_kernel_layers, units=pair_kernel_units)(block_output)
        block_output = L.Concatenate(2)([features, block_output])
    block_output = ConvPair(layers=pair_kernel_layers, units=pair_kernel_units)(block_output)
    return L.Add()([block_output, noise_or_output])


def make_resnet(name, d_in, d_out, compressor_kernel_layers, compressor_kernel_units, pair_kernel_layers, pair_kernel_units, blocks, block_layers):
    features = K.Input((None, d_in))
    noise = K.Input((None, d_out))
    compressed_features = ConvKernel(layers=compressor_kernel_layers, units=compressor_kernel_units)(features)
    output = make_block(compressed_features, noise, block_layers, pair_kernel_layers, pair_kernel_units)
    for i in range(1, blocks):
        output = make_block(compressed_features, output, block_layers, pair_kernel_layers, pair_kernel_units)
    output *= -1
    resnet = K.Model([features, noise], output)
    # K.utils.plot_model(resnet, name + '.png', show_shapes=True)
    return resnet
# end model


# begin dataset
def parse_feature(byte_string, d):
    tensor = tf.io.parse_tensor(byte_string, tf.float32)
    return tf.reshape(tensor, [-1, d])


def parse_example(example):
    context_features = {'protein': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = {
        'initial_positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
        'positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
        'features': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
        'masses': tf.io.FixedLenSequenceFeature([], dtype=tf.string)
    }
    context, sequence = tf.io.parse_single_sequence_example(example, context_features=context_features, sequence_features=sequence_features)
    initial_positions = parse_feature(sequence['initial_positions'][0], 3)
    features = parse_feature(sequence['features'][0], 9)
    masses = parse_feature(sequence['masses'][0], 1)
    initial_distances = calculate_distances(initial_positions)
    # protein_name_tensor = context['protein']
    positions = parse_feature(sequence['positions'][0], 3)
    features = tf.concat([masses, features], 1)
    masses = tf.concat([masses, masses, masses], 1)
    return initial_positions, initial_distances, positions, masses, features


def read_dataset():
    path = os.path.join('.', 'tfrecords')
    recordpaths = []
    for name in os.listdir(path):
        recordpaths.append(os.path.join(path, name))
    dataset = tf.data.TFRecordDataset(recordpaths, 'ZLIB')
    dataset = dataset.shuffle(420)
    dataset = dataset.map(map_func=parse_example)
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    return iterator
# end dataset


# begin step / loss
def step(initial_positions, initial_distances, positions, velocities,
         masses, num_atoms, num_atoms_squared, force_field):
    positions, velocities = move_atoms(positions, velocities, masses, force_field)
    loss_value = loss(initial_positions, initial_distances, positions,
                      velocities, masses,
                      num_atoms, num_atoms_squared, force_field)
    return positions, velocities, loss_value


def move_atoms(positions, velocities, masses, force_field):
    acceleration = force_field / masses
    noise = tf.random.normal(acceleration.shape, 0, NOISE, dtype='float32')
    velocities += acceleration + noise
    positions += velocities
    return positions, velocities


def loss(initial_positions, initial_distances, positions, velocities, masses, num_atoms, num_atoms_squared, force_field):
    # position
    num_atoms = tf.dtypes.cast(num_atoms, dtype = tf.float32)
    position_error = K.losses.mean_squared_error(initial_positions, positions)
    position_error /= num_atoms
    position_error *= POSITION_ERROR_WEIGHT
    # shape
    distances = calculate_distances(positions)
    shape_error = K.losses.mean_squared_error(initial_distances, distances)
    shape_error /= num_atoms_squared
    shape_error *= SHAPE_ERROR_WEIGHT
    # action energy
    kinetic_energy = (masses / 2) * (velocities ** 2)
    potential_energy = -1 * force_field
    average_action_error = tf.reduce_mean(kinetic_energy - potential_energy)
    average_action_error *= ACTION_ERROR_WEIGHT
    return position_error + shape_error + average_action_error


def calculate_distances(positions):
    positions = tf.squeeze(positions)
    distances = tf.reduce_sum(positions * positions, 1)
    distances = tf.reshape(distances, [-1, 1])
    return distances - 2 * tf.matmul(positions, tf.transpose(positions)) + tf.transpose(distances)
# end step/loss


def run_episode(adam, agent, iterator):
    done, train_step = False, 0
    batch_stop_loss_condition, batch_loss_value = [], []
    batch_positions, batch_features, batch_velocities =[], [], []
    batch_initial_positions, batch_initial_distances, batch_masses, batch_num_atoms = [], [], [], []
    for i in range(ITERATOR_BATCH):
        protein_data = iterator.get_next()
        initial_positions, initial_distances, positions, masses, features = protein_data
        batch_positions.append(positions), batch_features.append(features), batch_initial_positions.append(initial_positions)
        batch_initial_distances.append(initial_distances), batch_masses.append(masses)
        [print(i.shape) for i in protein_data]
        num_atoms = positions.shape[1].value
        batch_num_atoms.append(num_atoms)
        num_atoms_squared = num_atoms ** 2
        velocities = tf.random.normal(shape=positions.shape)
        batch_velocities.append(velocities)
        force_field = tf.zeros_like(velocities)
        initial_loss = loss(initial_positions, initial_distances, positions, velocities, masses, num_atoms, num_atoms_squared, force_field)
        num_atoms = initial_positions.shape[1]
        stop_loss = initial_loss * STOP_LOSS_MULTIPLIER
        stop_loss_condition = tf.reduce_mean(stop_loss, axis = -1)
        batch_stop_loss_condition.append(stop_loss_condition)
        stop_loss_condition = tf.reduce_mean(batch_stop_loss_condition, axis = -1)
    while not done:
        with tf.GradientTape() as tape:
            for i in range(ITERATOR_BATCH):
                positions, velocities, features = batch_positions[i], batch_velocities[i], batch_features[i]
                masses, num_atoms = batch_masses[i], batch_num_atoms[i]
                atoms = tf.concat([positions, velocities, features], 2)
                # atoms = tf.expand_dims(tf.concat([positions, velocities, features], 1), 0)
                noise = tf.expand_dims(tf.random.normal((num_atoms, 3)), 0)
                force_field = agent([atoms, noise])
                # force_field = tf.squeeze(agent([atoms, noise]), 0)
                initial_positions, initial_distances, positions = batch_initial_positions[i], batch_initial_distances[i], batch_positions[i]
                num_atoms_squared = num_atoms**2
                positions, velocities, loss_value = step(initial_positions, initial_distances, positions, velocities, masses, num_atoms, num_atoms_squared, force_field)
                batch_loss_value.append(tf.reduce_sum(loss_value, axis = 1))
        batch_loss_value = tf.convert_to_tensor(batch_loss_value)
        loss_value = tf.reduce_sum(batch_loss_value, axis = 1)
        gradients = tape.gradient(loss_value, agent.trainable_weights)
        step_mean_loss = tf.reduce_mean(loss_value, axis = -1)
        print('step', train_step, 'mean loss', step_mean_loss.numpy())
        print("stop_loss_condition", stop_loss_condition)
        stop_loss_condition = tf.reduce_mean(stop_loss_condition, axis = -1)
        done_because_loss = step_mean_loss > stop_loss_condition
        train_step += 1
        done_because_step = train_step > N_STEPS
        done = done_because_step or done_because_loss
        if not done:
            current_stop_loss = loss_value * STOP_LOSS_MULTIPLIER
            current_stop_loss_condition = tf.reduce_mean(current_stop_loss, axis = -1)
            if current_stop_loss_condition.numpy().item() < stop_loss_condition.numpy().item():
                stop_loss_condition = current_stop_loss_condition
                print('new stop loss:', current_stop_loss_condition.numpy())
    adam.apply_gradients(zip(gradients, agent.trainable_weights))
    reason = 'STEP' if done_because_step else 'STOP LOSS'
    print('done because of', reason)
    initial_loss = tf.reduce_mean(initial_loss, axis = 1)
    loss_value = tf.reduce_mean(loss_value, axis = 1)
    percent_improvement = ((initial_loss - loss_value) / initial_loss ) * 100
    print('improved by', percent_improvement.numpy(), '%')


@skopt.utils.use_named_args(dimensions=dimensions)
def train(compressor_kernel_layers,
          compressor_kernel_units,
          pair_kernel_layers,
          pair_kernel_units,
          blocks,
          block_layers,
          learning_rate,
          decay):
    try:
        start = time.time()
        TIME = str(start)
        iterator = read_dataset()
        agent = make_resnet('agent', 16, 3, compressor_kernel_layers, compressor_kernel_units, pair_kernel_layers, pair_kernel_units, blocks, block_layers)
        adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=decay)
        cumulative_improvement, episode, iterator_is_done = 0, 0, False
        while not iterator_is_done:
            print('')
            print('model', TIME, 'episode', episode)
            print("compressor_kernel_layers", compressor_kernel_layers,
                  "compressor_kernel_units", compressor_kernel_units,
                  "pair_kernel_layers", pair_kernel_layers,
                  "pair_kernel_units", pair_kernel_units,
                  "blocks", blocks,
                  "block_layers", block_layers,
                  "learning_rate", learning_rate,
                  "decay", decay)
            try:
                actual_cumulative_improvement, iterator_is_done = run_episode(adam, agent, iterator)
                cumulative_improvement += actual_cumulative_improvement
            except Exception as e:
                print(e)
            episode += 1
        cumulative_improvement /= N_EPISODES
        global best_cumulative_improvement
        if cumulative_improvement > best_cumulative_improvement:
            print('new best mean cumulative improvement:', cumulative_improvement.numpy())
            best_cumulative_improvement = cumulative_improvement
            run_path = os.path.join('.', 'runs', TIME)
            os.makedirs(run_path)
            save_path = os.path.join(run_path, 'best_agent.h5')
            tf.saved_model.save(agent, save_path)
            print('saved agent to', save_path)
        K.backend.clear_session()
        del agent
        return -1 * cumulative_improvement.numpy()[0]
    except Exception as e:
        print(e)
        return 1000


def main():
    search_result = skopt.gp_minimize(func=train,
                            dimensions=dimensions,
                            acq_func='EI',
                            x0=default_hyperparameters)

    print(search_result)


if __name__ == '__main__':
    main()

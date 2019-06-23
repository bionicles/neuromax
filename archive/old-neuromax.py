# neuromax.py - why?: 1 simple file with functions over classes
import tensorflow.keras.backend as B
import tensorflow.keras.layers as L
import tensorflow.keras as K
import tensorflow as tf
import random
import skopt
import time
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
BATCH_SIZE = 2
BATCH_SIZE = 2
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
# more ideas
# shake-shake dropconnect true/false (use dropout on weight each step) incl shakedrop (negative weights!)
# different activations
# tensorflow probability layers (bayesian deep learning)
# adam optimizer epsilon value
# normalization ... batch or layer norm
# stochastic depth in resnet
# how do you use attention?
# how do you use meta learning?

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
    128,
    2,
    128,
    2,
    2,
    0.001,
    0.0001
]


# begin model
class ShakeDropConnectDense(L.Dense):
    def call(self, inputs):
        if random.random() > 0.5:
            kernel = B.dropout(self.kernel, 0.5) * random.uniform(-1, 1)
        else:
            kernel = self.kernel
        outputs = B.dot(inputs, kernel)
        return self.activation(outputs)


def get_layer(units):
    return ShakeDropConnectDense(units, activation='tanh', use_bias=False)


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
    # resnet.summary()
    return resnet
# end model


# begin dataset
def parse_protein(example):
    context_features = {'protein': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = {
        'initial_positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
        'positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
        'features': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
        'masses': tf.io.FixedLenSequenceFeature([], dtype=tf.string)
    }
    context, sequence = tf.io.parse_single_sequence_example(example, context_features=context_features, sequence_features=sequence_features)
    initial_positions = tf.reshape(tf.io.parse_tensor(sequence['initial_positions'][0], tf.float32), [-1, 3])
    features = tf.reshape(tf.io.parse_tensor(sequence['features'][0], tf.float32), [-1, 9])
    masses = tf.reshape(tf.io.parse_tensor(sequence['masses'][0], tf.float32), [-1, 1])
    positions = tf.reshape(tf.io.parse_tensor(sequence['positions'][0], tf.float32), [-1, 3])
    features = tf.concat([masses, features], 1)
    masses = tf.concat([masses, masses, masses], 1)
    return initial_positions, positions, masses, features


def make_get_batch_function(protein_dataset):
    def get_batch():
        batch_initial_positions, batch_initial_distances, batch_masses, batch_num_atoms = [], [], [], []
        batch_stop_loss_condition, batch_loss_value, batch_step_mean_loss = [], [], []
        batch_positions, batch_features, batch_velocities =[], [], []
        batch_of_protein_tuples = protein_dataset.take(BATCH_SIZE)
        for protein in batch_of_protein_tuples:
            initial_positions, positions, masses, features = protein
            initial_distances = calculate_distances(initial_positions)
            batch_positions.append(positions), batch_features.append(features), batch_initial_positions.append(initial_positions)
            batch_initial_distances.append(initial_distances), batch_masses.append(masses)
            [print(i.shape) for i in protein]
            num_atoms = positions.shape[1].value
            batch_num_atoms.append(num_atoms)
            velocities = tf.random.normal(shape=positions.shape)
            batch_velocities.append(velocities)
            force_field = tf.zeros_like(velocities)
            initial_loss = loss(initial_positions, initial_distances, positions, velocities, masses, num_atoms, force_field)
            num_atoms = initial_positions.shape[1]
            stop_loss = initial_loss * STOP_LOSS_MULTIPLIER
            stop_loss_condition = tf.reduce_mean(stop_loss, axis = -1)
            batch_stop_loss_condition.append(stop_loss_condition)
            stop_loss_condition = tf.reduce_mean(batch_stop_loss_condition, axis = 0)
        batch = (batch_initial_positions, batch_initial_distances, batch_positions, batch_velocities, batch_masses, batch_num_atoms, stop_loss_condition)
        return batch
    return get_batch


def read_shards():
    path = os.path.join('.', 'tfrecords')
    recordpaths = []
    for name in os.listdir(path):
        recordpaths.append(os.path.join(path, name))
    dataset = tf.data.TFRecordDataset(recordpaths, 'ZLIB')
    dataset = dataset.shuffle(420)
    dataset = dataset.map(map_func=parse_protein)
    dataset = dataset.prefetch(buffer_size=BATCH_SIZE)
    return dataset


def get_batch_dataset(protein_dataset):
    get_batch = make_get_batch_function(protein_dataset)
    return tf.data.Dataset.from_generator(get_batch).prefetch(buffer_size=1)
# end dataset


# begin step / loss
@tf.function
def step(initial_positions, initial_distances, positions, velocities,
         masses, num_atoms, force_field):
    positions, velocities = move_atoms(positions, velocities, masses, force_field)
    loss_value = loss(initial_positions, initial_distances, positions,
                      velocities, masses, num_atoms, force_field)
    return positions, velocities, loss_value


@tf.function
def move_atoms(positions, velocities, masses, force_field):
    acceleration = force_field / masses
    noise = tf.random.normal(acceleration.shape, 0, NOISE, dtype='float32')
    velocities += acceleration + noise
    positions += velocities
    return positions, velocities


@tf.function
def loss(initial_positions, initial_distances, positions, velocities, masses, num_atoms, force_field):
    # position
    num_atoms = tf.dtypes.cast(num_atoms, dtype = tf.float32)
    position_error = K.losses.mean_squared_error(initial_positions, positions)
    position_error /= num_atoms
    position_error *= POSITION_ERROR_WEIGHT
    print("position error shape", position_error.shape)
    # shape
    distances = calculate_distances(positions)
    shape_error = K.losses.mean_squared_error(initial_distances, distances)
    shape_error /= num_atoms
    shape_error *= SHAPE_ERROR_WEIGHT
    print("shape error shape", shape_error.shape)
    # action energy
    kinetic_energy = (masses / 2) * (velocities ** 2)
    potential_energy = -1 * force_field
    average_action_error = tf.reduce_mean(kinetic_energy - potential_energy)
    average_action_error *= ACTION_ERROR_WEIGHT
    print("average_action_error shape", average_action_error.shape)
    return position_error + shape_error + average_action_error


@tf.function
def calculate_distances(positions):
    positions = tf.squeeze(positions)
    distances = tf.reduce_sum(positions * positions, 1)
    distances = tf.reshape(distances, [-1, 1])
    return distances - 2 * tf.matmul(positions, tf.transpose(positions)) + tf.transpose(distances)
# end step/loss

# begin training
def step_agent_on_protein:

def step_agent_on_batch():

    return done, percent_improvement

def run_episode(adam, agent, batch):

    return percent_improvement, True


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
        protein_dataset = read_shards()
        batches_dataset = get_batch_dataset(protein_dataset)
        agent = make_resnet('agent', 16, 3, compressor_kernel_layers, compressor_kernel_units, pair_kernel_layers, pair_kernel_units, blocks, block_layers)
        adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=decay)
        cumulative_improvement, episode, iterator_is_done = 0, 0, False
        for batch in dataset:
            print('')
            print('', TIME, 'episode', episode)
            print("compressor_kernel_layers", compressor_kernel_layers, "compressor_kernel_units", compressor_kernel_units, "pair_kernel_layers", pair_kernel_layers, "pair_kernel_units", pair_kernel_units, "blocks", blocks, "block_layers", block_layers, "learning_rate", learning_rate, "decay", decay)
            batch_initial_positions, batch_initial_distances, batch_positions, batch_velocities, batch_masses, batch_features, batch_num_atoms, stop_loss_condition = batch
            batch_step_mean_loss, batch_loss_value = [], []
            done = False
            for train_step in N_STEPS:
                with tf.GradientTape(persistent = True) as tape:
                    for i in range(BATCH_SIZE):
                        positions, velocities, features = batch_positions[i], batch_velocities[i], batch_features[i]
                        atoms = tf.concat([positions, velocities, features], 2)
                        noise = tf.random.normal(positions.shape)
                        force_field = agent([atoms, noise])
                        positions, velocities, loss_value = step(batch_initial_positions[i], batch_initial_distances[i], positions, velocities, batch_masses[i], batch_num_atoms[i], force_field)
                        batch_loss_value.append(loss_value)
                        gradients = tape.gradient(batch_loss_value[i], agent.trainable_weights)
                        adam.apply_gradients(zip(gradients, agent.trainable_weights))
                    batch_loss_value[i] = tf.reduce_mean(batch_loss_value[i], axis = -1)
                    batch_loss_value = tf.convert_to_tensor(batch_loss_value)
                    loss_value = tf.reduce_sum(batch_loss_value, axis = 1)
                    step_mean_loss = tf.reduce_mean(loss_value, axis = -1)
                    batch_step_mean_loss.append(step_mean_loss)
                    print('step', train_step, 'stop_loss_condition', stop_loss_condition.numpy(), 'mean loss', step_mean_loss.numpy())
                    done_because_loss = step_mean_loss > stop_loss_condition
                    done_because_step = train_step > N_STEPS
                    done = done_because_step or done_because_loss
                    if not done:
                        current_stop_loss = loss_value * STOP_LOSS_MULTIPLIER
                        current_stop_loss_condition = tf.reduce_mean(current_stop_loss, axis = -1)
                        # we update the trailing stop loss
                        if current_stop_loss_condition.numpy().item() < stop_loss_condition.numpy().item():
                            stop_loss_condition = current_stop_loss_condition
                            print('new stop loss:', current_stop_loss_condition.numpy())
                    else:
                        reason = 'STEP' if done_because_step else 'STOP LOSS'
                        print('done because of', reason)
                        initial_loss = tf.reduce_mean(initial_loss, axis = 1)
                        batch_step_mean_loss = tf.convert_to_tensor(batch_step_mean_loss)
                        all_mean_loss = tf.reduce_mean(batch_step_mean_loss, axis = -1)
                        batch_step_mean_loss = []
                        percent_improvement = ((initial_loss - all_mean_loss) / initial_loss ) * 100
                        percent_improvement = tf.reduce_sum(percent_improvement, axis = 0)
                        print('improved by', percent_improvement.numpy(), '%')
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
        B.clear_session()
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

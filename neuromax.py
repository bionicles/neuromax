# neuromax.py - why?: 1 simple file with functions over classes
from tensorflow.keras.layers import Layer, Dense, Concatenate
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras import Model, Input
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.util import load_logs
from bayes_opt.event import Events
from functools import partial
import numpy as np
import imageio
import random
import shutil
import time
import csv
import os
# globals
save_model = False
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
BIGGEST_FIRST_IF_NEG = 1
RANDOM_PROTEINS = True
N_PARALLEL_CALLS = 1
BATCH_SIZE = 1
# training
STOP_LOSS_MULTIPLIER = 1.04
NUM_RANDOM_TRIALS, NUM_TRIALS = 2, 3
PEDAGOGY_FILE_NAME = 'less-than-164kd-9-chains.csv'
NUM_EXPERIMENTS = 100
NUM_EPISODES = 42
NUM_STEPS = 100
VERBOSE = True
# hyperparameters
NUM_RANDOM_VALIDATION_TRIALS, NUM_VALIDATION_TRIALS, NUM_VALIDATION_EPISODES = 0, 1, 10000
COMPLEXITY_PUNISHMENT = 0  # 0 is off, higher is simpler
TIME_PUNISHMENT = 0
pbounds = {
    'BLOCKS': (1, 32),
    'LAYERS': (1, 8),
    'LR': (1e-4, 1e-1),
    'EPSILON': (1e-4, 1),
}

# begin model
def get_mlp(features, outputs, units_array):
    input = Input((features))
    output = Dense(units_array[0], activation='tanh', kernel_initializer=Orthogonal)(input)
    for units in units_array[1:]:
        output = Dense(units, activation='tanh', kernel_initializer=Orthogonal)(output)
    output = Dense(outputs, activation='tanh', kernel_initializer=Orthogonal)(output)
    return Model(input, output)


class ConvKernel(Layer):
    def __init__(self, units_array=[2048, 2048], features=16, outputs=5):
        super(ConvKernel, self).__init__()
        self.kernel = get_mlp(features, outputs, units_array)
        self.outputs = outputs

    def __call__(self, inputs):
        return tf.vectorized_map(self.kernel, inputs)

class ConvPair(Layer):
    def __init__(self,
                 units_array=[128, 128],
                 features=8,
                 outputs=3):
        super(ConvPair, self).__init__()
        self.kernel = get_mlp(features * 2, outputs, units_array)
        self.outputs = outputs

    def __call__(self, inputs):
        self.inputs = inputs
        return tf.vectorized_map(self.compute_atom, inputs)

    def compute_atom(self, atom1):
        def compute_pair(atom2):
            pair = tf.concat([atom1, atom2])
            return self.kernel(pair)
        contributions = tf.vectorized_map(compute_pair, self.inputs)
        return tf.reduce_sum(contributions)


def make_block(features, noise_or_output, n_layers):
    block_output = Concatenate(2)([features, noise_or_output])
    for layer_n in range(0, int(n_layers) - 1):
        block_output = ConvPair()(block_output)
        block_output = Concatenate(2)([features, block_output])
    block_output = ConvPair()(block_output)
    block_output = Add()([block_output, noise_or_output])
    return block_output


def make_resnet(name, d_in, d_out, blocks, layers):
    features = Input((None, d_in))
    noise = Input((None, d_out))
    compressed_features = ConvKernel()(features)
    output = make_block(compressed_features, noise, layers)
    for i in range(1, round(blocks.item())):
        output = make_block(compressed_features, noise, layers)
    output *= -1
    return Model([features, noise], output)
# end model

# begin dataset
def parse_feature(byte_string, d):
    tensor = tf.io.parse_tensor(byte_string, tf.float32)
    return tf.reshape(tensor, [-1, d])


def parse_example(example):
    context_features={'protein': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features={
        'initial_positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),  # num_atoms * 3
        'positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),  # num_atoms * 3
        'features': tf.io.FixedLenSequenceFeature([], dtype=tf.string),  # num_atoms * 16
        'masses': tf.io.FixedLenSequenceFeature([], dtype=tf.string)  # num_atoms * 3
    }
    context, sequence = tf.io.parse_single_sequence_example(example, context_features=context_features, sequence_features=sequence_features)
    initial_positions = parse_feature(sequence['initial_positions'][0], 3)
    features = parse_feature(sequence['features'][0], 9)
    masses = parse_feature(sequence['masses'][0], 1)
    return {
        'initial_positions': initial_positions,
        'initial_distances': calculate_distances(initial_positions),
        'protein_name_tensor': context['protein'],
        'positions': parse_feature(sequence['positions'][0], 3),
        'masses': tf.concat([masses, masses, masses], 1),
        'features': tf.concat([masses, features], 1),
        }


def make_dataset():
    path = os.path.join('.', 'tfrecords')
    recordpaths = []
    for name in os.listdir(path):
        recordpaths.append(os.path.join(path, name))
    print(recordpaths)
    dataset = tf.data.TFRecordDataset(recordpaths, 'ZLIB')
    dataset.map(map_func=parse_example,
                num_parallel_calls=N_PARALLEL_CALLS)
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=BATCH_SIZE)
    return dataset


def reset(dataset):
    example = dataset.take(1)
    state = parse_example(example)
    return (
        state['protein'],
        state['initial_positions'],
        state['initial_distances'],
        state['positions'],
        state['masses'],
        state['features'])
# end dataset


# begin step / loss
def step(initial_positions, initial_distances, positions, velocities, masses, num_atoms, num_atoms_squared, force_field):
    positions, velocities = move_atoms(positions, velocities, masses, force_field)
    loss_value = loss(initial_positions, initial_distances, positions, velocities, masses, force_field)
    return positions, velocities, loss_value

def move_atoms(positions, velocities, masses, force_field):
    acceleration = force_field / masses
    noise = tf.random.normal((num_atoms, 3), 0, NOISE, dtype='float16')
    velocities += acceleration + noise
    positions += velocities
    return positions, velocities

def loss(initial_positions, initial_distances, positions, velocities, masses, num_atoms, num_atoms_squared, force_field):
    # position
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
    distances = tf.reduce_sum(positions * positions, 1)
    distances = tf.reshape(distances, [-1, 1])
    return distances - 2 * tf.matmul(positions, tf.transpose(positions)) + tf.transpose(distances)
# end step/loss


def train(BLOCKS, LAYERS, LR, EPSILON):
    start = time.time()
    TIME = str(start)
    if save_model:
        run_path = os.path.join('.', 'runs', TIME)
        os.makedirs(run_path)
        save_path = os.path.join(run_path, 'model.h5')
    train_step = 0
    dataset = make_dataset()
    agent = make_resnet('agent', 17, 3, blocks=BLOCKS, layers=LAYERS)
    decayed_lr = tf.train.exponential_decay(LR, train_step, 10000, 0.96, staircase=True)
    adam = tf.train.AdamOptimizer(decayed_lr, epsilon=EPSILON)
    cumulative_improvement, episode = 0, 0
    for i in range(NUM_EPISODES):
        print('')
        print('BEGIN EPISODE', episode)
        done, step = False, 0
        protein, num_atoms, initial_positions, initial_distances, positions, velocities, masses, features = reset(iterator)
        initial_loss = loss(initial_positions, initial_distances, positions, velocities, masses)
        stop_loss = initial_loss * STOP_LOSS_MULTIPLIER
        step += 1
        while not done:
            print('')
            print('experiment', experiment, 'model', TIME, 'episode', episode, 'step', step)
            print('BLOCKS', round(BLOCKS), 'LAYERS', round(LAYERS), 'UNITS', round(UNITS), 'LR', LR)
            with tf.GradientTape() as tape:
                atoms = tf.expand_dims(tf.concat([positions, velocities, features], 1), 0)
                noise = tf.expand_dims(tf.random.normal((num_atoms, 3)), 0)
                force_field = tf.squeeze(agent([atoms, noise]), 0)
                positions, velocities, loss_value = step(initial_positions, initial_distances, positions, velocities, masses, force_field)
            gradients = tape.gradient(loss_value, agent.trainable_weights)
            adam.apply_gradients(zip(gradients, agent.trainable_weights))
            train_step += 1
            step += 1
            done_because_step = step > NUM_STEPS
            done_because_loss = loss_value > stop_loss
            done = done_because_step or done_because_loss
            if not done:
                current_stop_loss = loss_value * STOP_LOSS_MULTIPLIER
                stop_loss = current_stop_loss if current_stop_loss < stop_loss else stop_loss
        reason = 'STEP' if done_because_step else 'STOP LOSS'
        print('done because of', reason)
        percent_improvement = (initial_loss - loss_value) / 100
        cumulative_improvement += percent_improvement
        episode += 1
        if save_model:
            tf.saved_model.save(agent, save_path)
    if COMPLEXITY_PUNISHMENT is not 0:
        cumulative_improvement /= agent.count_params() * COMPLEXITY_PUNISHMENT
    if TIME_PUNISHMENT is not 0:
        elapsed = time.time() - start
        cumulative_improvement /= elapsed * TIME_PUNISHMENT
    cumulative_improvement /= NUM_EPISODES
    return cumulative_improvement


def trial(BLOCKS, LAYERS, LR, EPSILON):
    tf.keras.backend.clear_session()
    # try:
    return train(BLOCKS, LAYERS, LR, EPSILON)
    # except Exception as e:
    #     print('EXPERIMENT FAIL!!!', e)
    #     return -10


def main():
    global NUM_EPISODES, experiment, save_model
    print('eager?', tf.executing_eagerly())
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logger = JSONLogger(path='./runs/logs.json')
    bayes = BayesianOptimization(f=partial(trial),
                                 pbounds=pbounds,
                                 verbose=2,
                                 random_state=1)
    bayes.subscribe(Events.OPTMIZATION_STEP, logger)
    try:
        load_logs(bayes, logs=['./runs/logs.json'])
    except Exception as e:
        print('failed to load bayesian optimization logs', e)
    for e in range(NUM_EXPERIMENTS):
        experiment = e
        bayes.maximize(init_points=NUM_RANDOM_TRIALS, n_iter=NUM_TRIALS)
        print("BEST MODEL:", bayes.max)
    NUM_EPISODES, save_model = NUM_VALIDATION_EPISODES, True
    bayes.maximize(init_points=0, n_iter=1)
    for i, res in enumerate(bayes.res):
        print('iteration: {}, results: {}'.format(i, res))


if __name__ == '__main__':
    main()

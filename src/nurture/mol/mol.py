from datetime import datetime
import tensorflow as tf
from pymol import cmd, util
import numpy as np
import imageio
import random
import shutil
import gym
import os

from src.nurture.mol.make_dataset import load_proteins, load
from src.nurture.spaces import Ragged

B, L, K = tf.keras.backend, tf.keras.layers, tf.keras
CSV_FILE_NAME = 'sorted-less-than-256.csv'
TEMP_PATH = "../../archive/temp"
stddev = 273 * 0.001
DTYPE = tf.float32

class MolEnv(gym.Env):
    """Handle molecular folding and docking tasks."""

    def __init__(self, gifs=False):
        super(MolEnv, self).__init__()
        self.observation_space = Ragged(shape=(None, 10))
        self.action_space = Ragged(shape=(None, 3))
        self.protein_number = 0
        if gifs:
            self.pedagogy = load_proteins(CSV_FILE_NAME)
            self.reset = self.reset_gif
            self.step = self.step_gif
        else:
            self.proteins_in_dataset, self.dataset = self.read_shards("cif")
            self.reset = self.reset_tfrecord
            self.step = self.step_tfrecord

    def get_loss(self):
        return tf.keras.losses.MAE(self.target, get_distances(self.current,
                                                              self.current))

    # @tf.function
    def reset_tfrecord(self):
        self.protein_number = self.protein_number + 1
        id_string, n_atoms, target, positions, features, masses = self.dataset.take(1).__iter__().__next__()
        print(id_string, n_atoms, target, positions, features, masses)
        self.common_prepare(id_string, n_atoms, target, positions, features, masses)
        return self.current

    # @tf.function
    def common_prepare(self, id_string, n_atoms, target, positions, features, masses):
        self.target = get_distances(target, target)
        self.velocities = tf.zeros_like(positions)
        self.id_string = id_string
        self.n_atoms = n_atoms
        self.positions = positions
        self.features = features
        self.masses = masses
        self.current = tf.concat([positions, features], -1)
        self.initial_loss = self.get_loss()
        self.initial_loss = tf.reduce_sum(self.initial_loss)
        self.stop = self.initial_loss * 1.2

    # @tf.function
    def step_tfrecord(self, forces):
        loss, done, change = self.common_step(forces)
        return self.current, loss, done, change

    def reset_gif(self):
        self.protein_number = self.protein_number + 1
        id = random.choice(self.pedagogy)
        id_string, n_atoms, target, positions, features, masses = self.parse_item(load("cif", id, screenshot=True))
        self.common_prepare(id_string, n_atoms, target, positions, features, masses)
        self.prepare_pymol()
        return self.current

    # @tf.function
    def common_step(self, forces):
        forces = forces / self.masses
        self.velocities = self.velocities + forces
        noise = tf.random.truncated_normal(tf.shape(self.positions), stddev=stddev)
        self.positions = self.positions + self.velocities + noise
        self.current = tf.concat([self.positions, self.features], -1)
        loss = self.get_loss()
        loss = tf.reduce_sum(loss)
        new_stop = loss * 1.2
        if new_stop < self.stop:
            self.stop = new_stop
        elif loss > self.stop or loss != loss:
            done = True
        change = ((loss - self.initial_loss) / self.initial_loss) * 100.
        return loss, done, change

    def step_gif(self, forces):
        loss, done, change = self.common_step(forces)
        self.move_atoms()
        take_screenshot()
        if done:
            self.make_gif()
        return self.current, loss, done, change

    @tf.function
    def parse_item(self, example):
        context_features = {'id': tf.io.FixedLenFeature([], dtype=tf.string)}
        sequence_features = {'target': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                             'positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                             'features': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                             'masses': tf.io.FixedLenSequenceFeature([], dtype=tf.string)}
        context, sequence = tf.io.parse_single_sequence_example(example, context_features=context_features, sequence_features=sequence_features)
        target = tf.reshape(tf.io.parse_tensor(sequence['target'][0], tf.float32), [-1, 10])
        positions = tf.reshape(tf.io.parse_tensor(sequence['positions'][0], tf.float32), [-1, 3])
        features = tf.reshape(tf.io.parse_tensor(sequence['features'][0], tf.float32), [-1, 7])
        masses = tf.reshape(tf.io.parse_tensor(sequence['masses'][0], tf.float32), [-1, 1])
        masses = tf.concat([masses, masses, masses], 1)
        n_atoms = tf.shape(positions)[0]
        id_string = context['id']
        return id_string, n_atoms, target, positions, features, masses

    def read_shards(self, datatype):
        print("read_shards", datatype)
        dataset_path = os.path.join('.', 'src', 'nurture', 'mol', 'datasets', 'tfrecord', datatype)
        n_records = len(os.listdir(dataset_path))
        filenames = [os.path.join(dataset_path, str(i) + '.tfrecord') for i in range(n_records)]
        dataset = tf.data.TFRecordDataset(filenames, 'ZLIB')
        dataset = dataset.map(map_func=self.parse_item, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(1)
        return n_records, dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def move_atom(self, xyz):
        xyz = xyz.tolist()
        cmd.translate(xyz, f'id {str(self.atom_index)}')
        self.atom_index += 1

    def move_atoms(self):
        self.atom_index = 0
        np.array([self.move_atom(xyz) for xyz in self.velocities])

    def make_gif(self):
        imagepaths, images = [], []
        for filename in os.listdir(TEMP_PATH):
            print('processing', filename)
            filepath = os.path.join(TEMP_PATH, filename)
            imagepaths.append(filepath)
        imagepaths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for imagepath in imagepaths:
            image = imageio.imread(imagepath)
            images.append(image)
        now = str(datetime.datetime.now()).replace(" ", "_")
        gif_name = f'{self.protein_number}-{self.id_string}-{now}.gif'
        gif_path = os.path.join("..", "..", "archive", "gifs", gif_name)
        print('saving gif to', gif_path)
        imageio.mimsave(gif_path + self.id_string + ".gif", images)
        shutil.rmtree(TEMP_PATH)


def take_screenshot(step):
    print("")
    screenshot_path = f"{TEMP_PATH}/{step}.png"
    cmd.ray()
    # cmd.zoom("all")
    cmd.center("all")
    cmd.png(screenshot_path)


def prepare_pymol():
    cmd.clip("near", 100000)
    cmd.clip("slab", 100000)
    cmd.clip("far", -100000)
    cmd.show("surface")
    cmd.unpick()
    util.cbc()


@tf.function
def get_distances(a, b):  # L2
    print("tracing get_distances")
    return B.sqrt(B.sum(B.square(tf.expand_dims(a, 0) - tf.expand_dims(b, 1)), axis=-1))

    # @tf.function
    # def get_distances(a, b):  # L1
    #     print("tracing get_distances")
    #     return B.sum(B.abs(tf.expand_dims(a, 0) - tf.expand_dims(b, 1)), axis=-1))

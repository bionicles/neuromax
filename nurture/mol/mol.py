from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import tensorflow as tf
from pymol import cmd, util
import numpy as np
import imageio
import random
import shutil
import gym
import os

from src.nurture.mol.make_dataset import load, load_proteins
from src.nurture.spaces import Ragged

K = tf.keras
B, L = K.backend, K.layers

CSV_FILE_NAME = 'sorted-less-than-256.csv'
TEMP_PATH = "archive/temp"
stddev = 273 * 0.001
DTYPE = tf.float32


@tf.function
def parse_item(example):
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


def read_shards(datatype):
    print("read_shards", datatype)
    dataset_path = os.path.join('.', 'src', 'nurture', 'mol', 'datasets', 'tfrecord', datatype)
    n_records = len(os.listdir(dataset_path))
    filenames = [os.path.join(dataset_path, str(i) + '.tfrecord') for i in range(n_records)]
    dataset = tf.data.TFRecordDataset(filenames, 'ZLIB')
    dataset = dataset.map(map_func=parse_item, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(1)
    dataset = dataset.shuffle(n_records)
    return n_records, dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


@tf.function
def get_distances(a, b):  # L2
    a, b = tf.squeeze(a, 0), tf.squeeze(b, 0)
    return B.sum(B.square(tf.expand_dims(a, 0) - tf.expand_dims(b, 1)), axis=-1)


def get_loss(target_distances, current):
    current = get_distances(current, current)
    return tf.keras.losses.MAE(target_distances, current)


def get_trainer(agent, optimizer):
    @tf.function
    def run_mol_episode(element):
        print("tracing run_episode")
        current = tf.concat([positions, features], -1)
        target_distances = get_distances(target, target)
        total_forces = tf.zeros_like(positions)
        velocities = tf.zeros_like(positions)
        forces = tf.zeros_like(positions)
        initial_loss = 0.
        loss = 0.
        stop = 0.
        for step in tf.range(MAX_STEPS):
            with tf.GradientTape() as tape:
                forces = agent(current) / masses
                total_forces = total_forces + forces
                velocities = velocities + forces
                noise = tf.random.truncated_normal(
                    tf.shape(positions), stddev=(0.001 * TEMPERATURE))
                positions = positions + velocities + noise
                current = tf.concat([positions, features], -1)
                loss = get_loss(target_distances, positions)
            gradients = tape.gradient(
                [loss, tf.abs(total_forces), tf.abs(forces)],
                agent.trainable_weights)
            optimizer.apply_gradients(zip(gradients, agent.trainable_weights))
            ema.apply(agent.weights)
            loss = tf.reduce_sum(loss)
            new_stop = loss * 1.2
            if step < 1:
                initial_loss = loss
                stop = new_stop
            if new_stop < stop:
                stop = new_stop
            elif step > 0 and (loss > stop or loss != loss):
                break
        change = ((loss - initial_loss) / initial_loss) * 100.
        return change if not tf.math.is_nan(change) else 420.

    @tf.function
    def mol_trainer(dataset):
        print("tracing train")
        changes = []
        change = 0.
        for episode_number, element in dataset.enumerate():
            with tf.device('/gpu:0'):
                change = run_episode(element)
            tf.print(f'episode {episode_number + 1}
                     {change} % change in loss')
            tf.summary.scalar('change', change)
            changes = tf.concat([changes, [change]], -1)
            if episode >= EPISODES_PER_TASK - 1:
                break
        return changes

    return mol_trainer


class MolEnv(gym.Env):
    """Handle molecular folding and docking tasks."""

    def __init__(self):
        super(MolEnv, self).__init__()
        self.observation_space = Ragged(shape=(None, 10))
        self.action_space = Ragged(shape=(None, 3))
        self.pedagogy = load_proteins(CSV_FILE_NAME)
        self.protein_number = 0

    def reset(self):
        try:
            shutil.rmtree(TEMP_PATH)
        except Exception as e:
            print(e)
        os.mkdir(TEMP_PATH)
        self.protein_number = self.protein_number + 1
        no_protein = True
        while no_protein:
            try:
                id = random.choice(self.pedagogy)
                id_string, n_atoms, target, positions, features, masses = parse_item(load("cif", id, screenshot=True))
                no_protein = False
            except Exception as e:
                print(e)
        self.step_number = 1
        self.target = get_distances(target, target)
        self.velocities = tf.zeros_like(positions)
        self.id_string = id_string
        self.n_atoms = n_atoms
        self.positions = positions
        self.features = features
        self.masses = masses
        self.current = tf.concat([positions, features], -1)
        self.initial_loss = self.get_loss(self.current)
        self.initial_loss = tf.reduce_sum(self.initial_loss)
        self.stop = self.initial_loss * 1.2
        prepare_pymol()
        take_screenshot(self.step_number)
        self.features = tf.expand_dims(self.features, 0)
        self.step_number += 1
        return self.current

    def step(self, forces):
        accelerations = forces / self.masses
        self.velocities = self.velocities + accelerations
        noise = tf.random.truncated_normal(tf.shape(self.positions), stddev=stddev)
        self.positions = self.positions + self.velocities + noise
        self.current = tf.concat([self.positions, self.features], -1)
        loss = self.get_loss(tf.squeeze(self.current, 0))
        loss = tf.reduce_sum(loss)
        new_stop = loss * 1.2
        done = False
        if new_stop < self.stop:
            self.stop = new_stop
        elif loss > self.stop or loss != loss:
            done = True
        change = ((loss - self.initial_loss) / self.initial_loss) * 100.
        self.move_atoms()
        take_screenshot(self.step_number)
        if done:
            self.make_gif()
        self.step_number += 1
        return self.current, loss, done, change

    def get_loss(self, current):
        current = get_distances(current, current)
        return tf.keras.losses.MAE(self.target, current)

    def move_atom(self, xyz):
        xyz = xyz.tolist()
        cmd.translate(xyz, f'id {str(self.atom_index)}')
        self.atom_index += 1

    def move_atoms(self):
        self.atom_index = 0
        np.array([self.move_atom(xyz) for xyz in tf.squeeze(self.velocities, 0).numpy()])

    def make_gif(self):
        original_image_path = os.path.join(TEMP_PATH, "0.png")
        now = str(datetime.now()).replace(" ", "_")
        id_string = str(self.id_string.numpy().decode())
        for f in os.listdir(TEMP_PATH):
            if f == "0.png":
                continue
            path = os.path.join(TEMP_PATH, f)
            images_list = [original_image_path, path]
            imgs = [Image.open(i) for i in images_list]
            min_img_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
            img = np.hstack((np.asarray(i.resize(min_img_shape, Image.ANTIALIAS)) for i in imgs ))
            img = Image.fromarray(img)
            x, y = img.size
            draw = ImageDraw.Draw(img)
            pdb_id_string = "PDB: " + id_string
            model_id_string = "Bit Pharma Neuromax: " + now
            x2, y2 = draw.textsize(model_id_string)
            x3, y3 = draw.textsize(pdb_id_string)
            draw.text(((x-x2) / 2, 10), model_id_string, (255, 255, 255))
            draw.text(((x-x3) / 2, y - 42), pdb_id_string, (255, 255, 255))
            img.save(os.path.join(TEMP_PATH, f))
        dirfiles = []
        images = []
        for filename in os.listdir(TEMP_PATH):
            if filename[-3:] == "png":
                dirfiles.append(filename)
        dirfiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for dirfile in dirfiles[1:]:
            filepath = os.path.join(TEMP_PATH, dirfile)
            images.append(imageio.imread(filepath))
        gif_name = f'{now}-{self.protein_number}-{id_string}.gif'
        gif_path = os.path.join("archive", "gifs", gif_name)
        imageio.mimsave(gif_path, images)
        print("done generating gif at ", gif_path)
        shutil.rmtree(TEMP_PATH)

def take_screenshot(step):
    screenshot_path = f"{TEMP_PATH}/{step}.png"
    cmd.zoom("all")
    cmd.png(screenshot_path, width=256, height=256)


def prepare_pymol():
    cmd.clip("near", 100000)
    cmd.clip("slab", 100000)
    cmd.clip("far", -100000)
    cmd.show("spheres")
    cmd.unpick()
    util.cbc()


@tf.function
def get_distances(a, b):  # L2
    return B.sum(B.square(tf.expand_dims(a, 0) - tf.expand_dims(b, 1)), axis=-1)

    # @tf.function
    # def get_distances(a, b):  # L1
    #     print("tracing get_distances")
    #     return B.sum(B.abs(tf.expand_dims(a, 0) - tf.expand_dims(b, 1)), axis=-1))

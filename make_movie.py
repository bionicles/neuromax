import tensorflow as tf
from pymol import cmd, util, movie
import random
import numpy as np
import time
import os
import shutil
import imageio
axis = ['x', 'y', 'z']
IMAGE_SIZE = 256
def move_atom(xyz, atom_index):
    atom_selection_string = 'id ' + str(atom_index)
    xyz = xyz.numpy().tolist()
    cmd.translate(xyz, atom_selection_string)
    atom_index += 1

def translate(force):
    force = tf.squeeze(force, axis=0)
    force = tf.transpose(force)
    model = cmd.get_model('all', 1)
    masses = np.array([atom.get_mass() for atom in model.atom])
    masses = [masses, masses, masses]
    velocity = force / masses
    velocity = tf.transpose(velocity)*100
    atom_index = 0
    for xyz in velocity:
        move_atom(xyz, atom_index)
        atom_index += 1

def get_positions():
    model = cmd.get_model('all', 1)
    positions = np.array(model.get_coord_list())
    return tf.convert_to_tensor(positions, preferred_dtype=tf.float32)

def get_atom_features(atom):
    return np.array([ord(atom.chain.lower())/122,
                     atom.formal_charge,
                     atom.partial_charge,
                     atom.vdw,
                     atom.b,
                     atom.get_free_valence(0),
                     sum([ord(i) for i in atom.resi]) / len(atom.resi),
                     sum([ord(i) for i in atom.resn]) / len(atom.resn),
                     sum([ord(i) for i in atom.symbol])/ len(atom.symbol)])

def prepare_pymol(pdb_name):
    # setup PyMOL for movies
    cmd.reinitialize()
    if not os.path.exists("pdbs/" + pdb_name + ".pdb"):
        print('fetching', pdb_name)
        cmd.fetch(pdb_name, path=os.path.join('.', 'pdbs'), type='pdb')
    cmd.load("pdbs/" + pdb_name+'.pdb')
    cmd.remove("solvent")
    util.cbc()
    cmd.set('max_threads', 16)
    cmd.show("surface")
    chains = cmd.get_chains()
    for chain in chains:
        cmd.extract(chain + chain, 'c. '+chain)
    cmd.remove("all and not (alt '')")  # remove alternate conformations
    return chains

def unfold_index(name, index):
    selection_string_array = [
        'first (({}`{}) extend 2 and name C)'.format(name, index),  # prev C
        'first (({}`{}) extend 1 and name N)'.format(name, index),  # this N
        '({}`{})'.format(name, index),                              # this CA
        'last (({}`{}) extend 1 and name C)'.format(name, index),   # this C
        'last (({}`{}) extend 2 and name N)'.format(name, index)]   # next N
    try:
        cmd.set_dihedral(selection_string_array[0],
                         selection_string_array[1],
                         selection_string_array[2],
                         selection_string_array[3], random.randint(0, 360))
        cmd.set_dihedral(selection_string_array[1],
                         selection_string_array[2],
                         selection_string_array[3],
                         selection_string_array[4], random.randint(0, 360))
    except Exception as e:
        print('failed to set dihedral at ', name, index)
        print(e)

def unfold():
    np.array([unfold_index(name, index) for name, index in
              cmd.index('byca (chain {})'.format('AA'))])

def make_gif(pdb_name):
    gif_name = '{}.gif'.format(pdb_name)
    gif_path = os.path.join("gifs", gif_name)
    imagepaths, images = [], []
    pngs_path = "./pngs"
    gif_path = "./gifs/"
    if not os.path.exists(pngs_path):
        os.makedirs(pngs_path)
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    for stackname in os.listdir(pngs_path):
        print('processing', stackname)
        filepath = os.path.join(pngs_path, stackname)
        imagepaths.append(filepath)
    imagepaths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for imagepath in imagepaths:
        image = imageio.imread(imagepath)
        images.append(image)
    print('saving gif to', gif_path)
    imageio.mimsave(gif_path + pdb_name + ".gif", images)
    shutil.rmtree(pngs_path)

def generate_movie(movie_name, pdb_name, agent, num_steps_gif):
    chains = prepare_pymol(pdb_name)
    for chain in chains:
        translation = [np.random.randint(-50, 50), np.random.randint(-50, 50), np.random.randint(-50, 50)]
        cmd.translate(translation, object=chain+chain)
    for chain in chains:
        rotation = [np.random.randint(-50, 50), np.random.randint(-50, 50), np.random.randint(-50, 50)]
        for i in range(3):
            cmd.rotate(axis[i], rotation[i], object=chain+chain)
    unfold()
    num_steps_gif = 3
    step = 0
    model = cmd.get_model('all', 1)
    features = np.array([get_atom_features(atom) for atom in model.atom])
    features = tf.dtypes.cast(features, dtype = tf.float32)
    masses = np.array([atom.get_mass() for atom in model.atom])
    masses = tf.expand_dims(masses, axis=1)
    masses = tf.dtypes.cast(masses, dtype = tf.float32)
    features = tf.concat([masses, features], -1)
    features = tf.expand_dims(features, axis=0)
    png_path = "pngs/screenshot"
    cmd.zoom("all")
    while (step<num_steps_gif):
        positions = get_positions()
        positions = tf.expand_dims(positions, axis=0)
        stacked_features = tf.concat([positions, features], axis=-1)
        compressed_noise = tf.random.truncated_normal((1, tf.shape(positions)[1], 5), stddev=0.01)
        output_noise = tf.random.truncated_normal(tf.shape(positions), stddev=0.01)
        forces = agent([stacked_features, compressed_noise, output_noise])
        translate(forces)
        cmd.png(png_path + str(step) + ".png", width=IMAGE_SIZE, height=IMAGE_SIZE)
        step += 1
    make_gif(pdb_name)

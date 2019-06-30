import tensorflow as tf
from pymol import cmd, util, movie
import random
import numpy as np
axis = ['x', 'y', 'z']

def move_atom(xyz):
    atom_selection_string = 'id ' + str(atom_index)
    xyz = xyz.numpy().tolist()
    cmd.translate(xyz, atom_selection_string)
    atom_index += 1

def translate(force):
    model = cmd.get_model('current', 1)
    masses = np.array([atom.get_mass() for atom in model.atom])
    velocity = force / masses
    atom_index = 0
    vectors = np.array([self.translate_atom(xyz) for xyz in velocity])

def translate_atom(vector):
    # print("translate_atom", vector, type(vector), vector.shape)
    movement_vector = list(vector) # list(movement_vector)
    atom_selection_string = "id " + str(atom_index)
    cmd.translate(movement_vector, atom_selection_string)
    atom_index += 1

def get_positions():
    model = cmd.get_model('all', 1)
    positions = np.array(model.get_coord_list())
    return tf.convert_to_tensor(positions, preferred_dtype=DTYPE)

def get_atom_features(atom):
    return np.array([ord(atom.chain.lower())/122,
                     atom.formal_charge,
                     atom.partial_charge,
                     atom.vdw,
                     atom.b,
                     atom.q,
                     atom.get_free_valence(0),
                     atom.resi,
                     atom.index,
                     sum([ord(i)/122 for i in atom.resn]) / len(atom.resn),
                     sum([ord(i)/122 for i in atom.symbol]) / len(atom.symbol),
                     sum([ord(i)/122 for i in atom.name]) / len(atom.name)])

def prepare_pymol(pdb_name):
    # setup PyMOL for movies
    cmd.reinitialize()
    cmd.set('matrix_mode', 1)
    cmd.set('movie_panel', 1)
    cmd.set('scene_buttons', 1)
    cmd.set('cache_frames', 1)
    cmd.config_mouse('three_button_motions', 1)
    if not os.path.exists("pdbs/" + pdb_name):
        print('fetching', protein)
        cmd.fetch(protein, path=os.path.join('.', 'pdbs'), type='pdb')
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

def generate_movie(length, movie_name, pdb_name, agent, start_after = 1):
    chains = prepare_pymol(pdb_name)
    cmd.mset("1x"+str(30*length))
    cmd.zoom("all", buffer=42, state=-1)
    cmd.frame(1)
    cmd.mview("store", object='all')
    cmd.frame(30*start_after) # start animation after start_after
    cmd.mview("store", object='all')
    cmd.frame(30*length*0.3)
    for chain in chains:
        translation = [np.random.randint(-50, 50), np.random.randint(-50, 50), np.random.randint(-50, 50)]
        cmd.translate(translation, object=chain+chain)
        cmd.mview('store', object='all')
    cmd.frame(30*length*0.3)
    for chain in chains:
        rotation = [np.random.randint(-50, 50), np.random.randint(-50, 50), np.random.randint(-50, 50)]
        for i in range(3):
            cmd.rotate(axis[i], rotation[i], object=chain+chain)
    cmd.mview('store', object='all')
    cmd.frame(30*2)
    unfold()
    cmd.mview('store', object='all')
    max_steps = 5
    step = 0
    features = np.array([get_atom_features(atom) for atom in model.atom])
    while (step<max_steps):
        positions = get_positions()
        stacked_features = tf.concat([positions, features], axis=-1)
        compressed_noise = tf.random.truncated_normal((1, tf.shape(positions)[1], 5), stddev=0.0001)
        output_noise = tf.random.truncated_normal(tf.shape(positions), stddev=0.0001)
        forces = agent([stacked_features, compressed_noise, output_noise])
        cmd.frame(30*length)
        translate(forces)
        step += 1
    movie.produce(filename = movie_name+'.mpg')

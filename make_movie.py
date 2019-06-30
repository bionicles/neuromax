from pymol import cmd, util, movie
import random
import tensorflow as tf
import numpy as np
axis = ['x', 'y', 'z']

def moves_atom(xyz):
    atom_selection_string = 'id ' + str(atom_index)
    xyz = xyz.numpy().tolist()
    cmd.translate(xyz, atom_selection_string)
    atom_index += 1

def prepare_pymol(pdb_name):
    # setup PyMOL for movies
    cmd.reinitialize()
    cmd.set('matrix_mode', 1)
    cmd.set('movie_panel', 1)
    cmd.set('scene_buttons', 1)
    cmd.set('cache_frames', 1)
    cmd.config_mouse('three_button_motions', 1)
    cmd.load(pdb_name+'.pdb')
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
    done = False
    while not done:
        forces = agent(pdb_featurs)
        cmd.frame(30*length)
        move_atoms(new_positions)
    movie.produce(filename = movie_name+'.mpg')

from pymol import cmd, util, movie
import random
import numpy as np
translation = [20, 20, 20]
rotation = [45, 45, 45]
axis = ['x', 'y', 'z']
pdb_name = '1a00'
def prepare_pymol():
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
    #cmd.show("spheres")
    chains = cmd.get_chains()
    undock_chain = random.choice(chains)
    cmd.extract('AA', 'c. '+undock_chain)
    cmd.remove("all and not (alt '')")  # remove alternate conformations

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

def generate_movie(length, movie_name, start_after = 1):
    prepare_pymol()
    cmd.mset("1x"+str(30*length))
    cmd.zoom("all", buffer=42, state=-1)
    cmd.frame(1)
    cmd.mview("store", object='AA')
    cmd.frame(30*start_after) # start animation after start_after
    cmd.mview("store", object='AA')
    cmd.frame(30*2) # four seconds of translation
    cmd.translate(translation, object='AA')
    cmd.mview('store', object='AA')
    cmd.frame(30*2) # 4 seconds of rotation
    for i in range(3):
        cmd.rotate(axis[i], rotation[i], object="AA")
    cmd.mview('store', object='AA')
    cmd.frame(30*2)
    unfold()
    cmd.mview('store', object='AA')
    cmd.frame(30*length)
    cmd.mview('interpolate', object='AA')
    movie.produce(filename = movie_name+'.mpg')

if __name__=='__main__':
    generate_movie(9, 'neuromax')

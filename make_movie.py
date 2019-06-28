from pymol import cmd, util, movie
import random
translation = [20, 20, 20]
rotation = [45, 45, 45]
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
    cmd.show("spheres")
    chains = cmd.get_chains()
    undock_chain = random.choice(chains)
    cmd.extract('AA', 'c. '+undock_chain)

def undock_protein(pdb_name, translation, rotation):
    cmd.translate(translation, object='AA')
    axis = ['x', 'y', 'z']
    #for i in range(3):
    #    cmd.rotate(axis[i], rotation[i], object="AA")
def generate_movie(length, movie_name, start_after = 1):
    prepare_pymol()
    cmd.mset("1x"+str(30*length))
    cmd.zoom("all", buffer=42, state=-1)
    cmd.frame(1)
    cmd.mview("store", object='AA')
    cmd.frame(30*start_after) # start animation after start_after 
    cmd.mview("store", object='AA')
    cmd.frame(30*length)
    undock_protein(pdb_name, translation, rotation)
    cmd.mview('store', object='AA')
    cmd.mview('interpolate', object='AA')
    movie.produce(filename = movie_name+'.mpg')

if __name__=='__main__':
    generate_movie(4, 'neuromax')

from pymol import movie, cmd, util
import numpy as np
import random


def unfold(chains):
    for chain in chains:
        np.array([unfold_index(name, index) for name, index in
                  cmd.index('byca (chain {})'.format(chain))])


def unfold_index(name, index):
    selection_string_array = [
        f'first (({name}`{index}) extend 2 and name C)',  # prev C
        f'first (({name}`{index}) extend 1 and name N)',  # this N
        f'({name}`{index})',                              # this CA
        f'last (({name}`{index}) extend 1 and name C)',   # this C
        f'last (({name}`{index}) extend 2 and name N)']   # next N
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


def make_movie(pdb, style):
    cmd.fetch(pdb)
    cmd.remove('solvent')
    cmd.remove("all and not (alt '')")  # remove alternate conformations
    util.cbc()
    cmd.mset("1x361")
    # undock and zoom
    translation = [random.randint(min, max) for _ in range(3)]
    rotation = [random.randint(0, 360) for _ in range(3)]
    axis = ['x', 'y', 'z']
    cmd.translate(translation, object='AA')
    [cmd.rotate(axis[i], rotation[i], object="AA") for i in range(3)]
    cmd.zoom("AA", buffer=42, state=-1)
    # reset object
    cmd.translate(translation * -1, object='AA')
    [cmd.rotate(axis[i], rotation[i] * -1, object="AA") for i in range(3)]
    # store original position keyframes
    cmd.mview('store', first=1, last=30, object='AA')
    cmd.mview('interpolate', first=91, last=361, object='AA')
    # undock and unfold
    cmd.translate(translation, object='AA')
    [cmd.rotate(axis[i], rotation[i], object="AA") for i in range(3)]
    unfold()
    cmd.mview('interpolate', first=31, last=91, object='AA')
    # make the movie
    movie.produce(filename=pdb + '.mpg')


def make_movies():
    pdbs = ['4LGP', '4KRM']
    styles = ['spheres', 'surface']

    for i in range(len(pdbs)):
        make_movie(pdbs[i], styles[i])


if __name__ == '__main__':
    make_movies()

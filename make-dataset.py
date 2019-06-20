from progiter import ProgIter
import tensorflow as tf
from pymol import cmd
import numpy as np
import random
import shutil
import csv
import os


tf.enable_eager_execution()


CSV_FILE_NAME = 'less-than-164kd-9-chains.csv'
MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE = 4, 64
PROTEINS_PER_TFRECORD = 8
DTYPE = tf.float32


def load_pedagogy(pedagogy_path):
    pedagogy = []
    with open(pedagogy_path) as csvfile:
        reader = csv.reader(csvfile)
        results = []
        for row in reader:
            for item in row:
                item = item.strip()
                results.append(item)
        pedagogy = results
    return pedagogy


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
                     atom.get_free_valence(0),
                     sum([ord(i) for i in atom.resi]) / len(atom.resi),
                     sum([ord(i) for i in atom.resn]) / len(atom.resn),
                     sum([ord(i) for i in atom.symbol])/ len(atom.symbol)])


def undock(chains):
    for chain in chains:
        selection_string = 'chain ' + chain
        translation_vector = [
            random.randrange(MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE),
            random.randrange(MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE),
            random.randrange(MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE)]
        cmd.translate(translation_vector, selection_string)
        cmd.rotate('x', random.randrange(-360, 360), selection_string)
        cmd.rotate('y', random.randrange(-360, 360), selection_string)
        cmd.rotate('z', random.randrange(-360, 360), selection_string)


def unfold(chains):
    for chain in chains:
        np.array([unfold_index(name, index) for name, index in
                  cmd.index('byca (chain {})'.format(chain))])


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


def load(protein):
    cmd.delete('all')
    pdb_file_name = protein + '.pdb'
    pdb_path = os.path.join('.', 'pdbs', pdb_file_name)
    print('')
    if not os.path.exists(pdb_path):
        print('fetching', protein)
        cmd.fetch(protein, path=os.path.join('.', 'pdbs'), type='pdb')
    elif os.path.exists(pdb_path):
        print('loading', pdb_path)
        cmd.load(pdb_path)
    cmd.remove("all and not (alt '')")  # remove alternate conformations
    cmd.remove('solvent')
    cmd.select(name='current', selection='all')
    initial_positions = get_positions()
    chains = cmd.get_chains('current')
    model = cmd.get_model('current', 1)
    features = np.array([get_atom_features(atom) for atom in model.atom])
    features = tf.convert_to_tensor(features, preferred_dtype=DTYPE)
    masses = np.array([atom.get_mass() for atom in model.atom])
    masses = tf.convert_to_tensor(masses, preferred_dtype=DTYPE)
    undock(chains)
    unfold(chains)
    positions = get_positions()
    return make_example(protein, initial_positions, positions, features, masses)


def make_example(protein, initial_positions, positions, features, masses):
    example = tf.train.SequenceExample()
    # non-sequential features
    example.context.feature["protein"].bytes_list.value.append(bytes(protein, 'utf-8'))
    # sequential features
    fl_initial_positions = example.feature_lists.feature_list["initial_positions"]
    fl_positions = example.feature_lists.feature_list["positions"]
    fl_features = example.feature_lists.feature_list["features"]
    fl_masses = example.feature_lists.feature_list["masses"]
    # sequential values
    fl_initial_positions.feature.add().bytes_list.value.append(tf.io.serialize_tensor(initial_positions).numpy())
    fl_positions.feature.add().bytes_list.value.append(tf.io.serialize_tensor(positions).numpy())
    fl_features.feature.add().bytes_list.value.append(tf.io.serialize_tensor(features).numpy())
    fl_masses.feature.add().bytes_list.value.append(tf.io.serialize_tensor(masses).numpy())
    # serialized
    return example.SerializeToString()


def write_shards():
    pedagogy_path = os.path.join('.', 'csvs', CSV_FILE_NAME)
    pedagogy = load_pedagogy(pedagogy_path)
    problems = []
    k = 0
    p = 0
    for protein in ProgIter(pedagogy, verbose=1):
        if p % PROTEINS_PER_TFRECORD is 0:
            try:
                writer.close()
            except Exception as e:
                print('writer.close() exception', e)
            shard_path = os.path.join('.', 'tfrecords', str(k) + '.tfrecord')
            writer = tf.io.TFRecordWriter(shard_path, 'ZLIB')
            k += 1
        try:
            data = load(protein)
            writer.write(data)
        except Exception as e:
            print('failed on', p, protein)
            print(e)
            problems.append([p, protein, e])
        print('wrote', protein, 'to', shard_path)
        p += 1
    print('problem children:')
    [print(problem) for problem in problems]
    print('done!')


def main():
    tfrecord_path = os.path.join('.', 'tfrecords')
    try:
        shutil.rmtree(tfrecord_path)
    except Exception as e:
        print('shutil.rmtree(tfrecord_path) exception', e)
    try:
        os.mkdir(tfrecord_path)
    except Exception as e:
        print('os.path.makedirs(tfrecord_path) exception', e)

    write_shards()



if __name__ == '__main__':
    main()

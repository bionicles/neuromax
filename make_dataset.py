from progiter import ProgIter
import tensorflow as tf
from pymol import cmd
import numpy as np
import random
import shutil
import csv
import os
import time
CSV_FILE_NAME = 'smallbig_less_then_9_chains.csv'
P_UNDOCK = 0.8
P_UNFOLD = 0.8
MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE = 4, 64
ITEMS_PER_SHARD = 8
DTYPE = tf.float32

elements = {'h': 1, 'he': 2, 'li': 3, 'be': 4, 'b': 5, 'c': 6, 'n': 7, 'o': 8,
'f': 9, 'ne': 10, 'na': 11, 'mg': 12, 'al': 13, 'si': 14, 'p': 15, 's': 16,
'cl': 17, 'ar': 18, 'k': 19, 'ca': 20, 'sc': 21, 'ti': 22, 'v': 23, 'cr': 24,
'mn': 25, 'fe': 26, 'co': 27, 'ni': 28, 'cu': 29, 'zn': 30, 'ga': 31, 'ge': 32,
'as': 33, 'se': 34, 'br': 35, 'kr': 36, 'rb': 37, 'sr': 38, 'y': 39, 'zr': 40,
'nb': 41, 'mo': 42, 'tc': 43, 'ru': 44, 'rh': 45, 'pd': 46, 'ag': 47, 'cd': 48,
'in': 49, 'sn': 50, 'sb': 51, 'te': 52, 'i': 53, 'xe': 54, 'cs': 55, 'ba': 56,
'la': 57, 'ce': 58, 'pr': 59, 'nd': 60, 'pm': 61, 'sm': 62, 'eu': 63, 'gd': 64,
'tb': 65, 'dy': 66, 'ho': 67, 'er': 68, 'tm': 69, 'yb': 70, 'lu': 71, 'hf': 72,
'ta': 73, 'w': 74, 're': 75, 'os': 76, 'ir': 77, 'pt': 78, 'au': 79, 'hg': 80,
'tl': 81, 'pb': 82, 'bi': 83, 'po': 84, 'at': 85, 'rn': 86, 'fr': 87, 'ra': 88,
'ac': 89, 'th': 90, 'pa': 91, 'u': 92, 'np': 93, 'pu': 94, 'am': 95, 'cm': 96,
'bk': 97, 'cf': 98, 'es': 99, 'fm': 100, 'md': 101, 'no': 102, 'lr': 103,
'rf': 104, 'db': 105, 'sg': 106, 'bh': 107, 'hs': 108, 'mt': 109, 'ds': 110,
'rg': 111, 'cn': 112}


def load_qm9():
    path = os.path.join('.', 'datasets', 'qm9')
    return os.listdir(path)


def load_rxns():
    path = os.path.join('.', 'datasets', 'rxn')
    return os.listdir(path)


def load_pdbs():
    with open(os.path.join('.', 'datasets', 'csvs', CSV_FILE_NAME)) as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        return [item.strip() for item in rows[0]]


def undock(chains, type):
    for chainOrName in chains:
        selection_string = f'chain {chainOrName}' if type is 'pdb' else chainOrName
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


def get_positions(model):
    positions = np.array(model.get_coord_list())
    return tf.convert_to_tensor(positions, dtype=DTYPE)


def get_features(model):
    features = np.array([get_atom_features(atom) for atom in model.atom])
    return tf.convert_to_tensor(features, dtype=DTYPE)


def get_atom_features(atom):
    resi = atom.resi
    try:
        resi = float(resi)
    except Exception as e:
        resi = float(resi[0:-1]) + float(ord(resi[-1])) / 122.
    return np.array([elements[atom.symbol.lower()],
                     atom.get_mass(),
                     atom.formal_charge,
                     atom.partial_charge,
                     atom.vdw,
                     atom.b,
                     atom.q,
                     atom.get_free_valence(0),
                     resi,
                     atom.index,
                     sum([ord(i) / 122 for i in atom.chain]),
                     sum([ord(i) / 122 for i in atom.resn]),
                     sum([ord(i) / 122 for i in atom.name])], dtype=np.float32)


def get_quantum_target(path):
    return tf.convert_to_tensor(open(path)[1].split("\t")[1:-1])


def get_reactants_products(path):
    line_is_n_reactants_products = False
    line_is_molfile = False
    mol_data = open(path)
    reactants = []
    products = []
    molfiles = []
    for line in mol_data:
        if line_is_n_reactants_products:
            reactants_products = line.split('\t')
            reactants = reactants_products[0]
            products = reactants_products[1]
        elif line_is_molfile:
            molfile = line.replace(':', '_')
            molfiles.append(f'{molfile}.mol')
        if "RHEA:release" in line:
            line_is_n_reactants_products = True
        if "$mol" in line:
            line_is_molfile = True
    for name in molfiles:
        if reactants > 0:
            reactants.push(name)
        else:
            products.push(name)
    return reactants, products


def clean_pymol():
    cmd.remove("all and not (alt '')")  # remove alternate conformations
    cmd.alter("all", 'alt=""')
    cmd.remove('solvent')


def load(type, id):
    # we clear pymol, make filename and path
    quantum_target = None
    target_features = None
    cmd.delete('all')
    file_name = f'{id}.{type}'
    dataset_path = os.path.join('.', 'datasets', type)
    path = os.path.join(dataset_path, file_name)
    # we load the target
    print(f'load {path}')
    if type is "pdb":  #bio
        if not os.path.exists(path):
            cmd.fetch(id, path=path, type='pdb')
        elif os.path.exists(path):
            cmd.load(path)
    elif type is "xyz":
        quantum_target = get_quantum_target(path)
        cmd.load(path)
    elif type is "rxn":
        reactants, products = get_reactants_products(path)
        [cmd.load(os.path.join('.', 'datasets', 'mol', product)) for product in products]
    # make target
    clean_pymol()
    model = cmd.get_model('all', 1)
    target_positions = get_positions(model)
    if type is 'rxn':
        target_features = get_features(model)
        cmd.delete('all')
        [cmd.load(os.path.join('.', 'datasets', 'mol', reactant)) for reactant in reactants]
        clean_pymol()
        model = cmd.get_model('all', 1)
    # make the model inputs
    if type is 'pdb':
        chains = cmd.get_chains('all')
        if random.random() < P_UNDOCK:
            undock(chains, type)
        if random.random() < P_UNFOLD:
            unfold(chains)
    if type is 'rxn':
        names = cmd.get_names('all')
        undock(names, type)
    positions = get_positions(model)
    features = get_features(model)
    return make_example(type, id, target_positions, positions, features, quantum_target, target_features)


def make_example(type, id, target_positions, positions, features, quantum_target=None, target_features=None):
    example = tf.train.SequenceExample()
    # non-sequential features
    example.context.feature["type"].bytes_list.value.append(bytes(type, 'utf-8'))
    example.context.feature["id"].bytes_list.value.append(bytes(id, 'utf-8'))
    if quantum_target:
        example.context.feature["quantum_target"].bytes_list.value.append(tf.io.serialize_tensor(quantum_target).numpy())
    # sequential features
    if target_features:
        fl_target_features = example.feature_lists.feature_list["target_positions"]
        fl_target_features.feature.add().bytes_list.value.append(tf.io.serialize_tensor(target_features).numpy())
    fl_target_positions = example.feature_lists.feature_list["target_positions"]
    fl_target_positions.feature.add().bytes_list.value.append(tf.io.serialize_tensor(target_positions).numpy())
    fl_positions = example.feature_lists.feature_list["positions"]
    fl_positions.feature.add().bytes_list.value.append(tf.io.serialize_tensor(positions).numpy())
    fl_features = example.feature_lists.feature_list["features"]
    fl_features.feature.add().bytes_list.value.append(tf.io.serialize_tensor(features).numpy())
    # serialized
    return example.SerializeToString()


def write_shards(qm9, rxns, pdbs):
    problems = []
    # write qm9 tfrecords
    for dataset, type in [(qm9, "xyz"), (rxns, "rxn"), (pdbs, "pdb")]:
        k, p = 0, 0
        for dataset_item in ProgIter(dataset, verbose=1):
            if p % ITEMS_PER_SHARD is 0:
                try:
                    writer.close()
                except Exception as e:
                    print('writer.close() exception', e)
                shard_path = os.path.join('.', 'datasets', 'tfrecord', type)
                if not os.path.exists(shard_path):
                    os.makedirs(shard_path)
                shard_path = os.path.join(shard_path, str(k) + '.tfrecord')
                writer = tf.io.TFRecordWriter(shard_path, 'ZLIB')
                k += 1
            try:
                print(type)
                time.sleep(5)
                data = load(type, dataset_item.lower())
                writer.write(data)
            except Exception as e:
                print('failed on', p, dataset_item)
                print(e)
                problems.append([p, dataset_item, e])
            print('wrote', dataset_item, 'to', shard_path)
            p += 1
    print('problem children:')
    [print(problem) for problem in problems]
    print('done!')


def main():
    tfrecord_path = os.path.join('.', 'datasets', 'tfrecord')
    try:
        shutil.rmtree(tfrecord_path)
    except Exception as e:
        print('shutil.rmtree(tfrecord_path) exception', e)
    try:
        os.mkdir(tfrecord_path)
    except Exception as e:
        print('os.path.makedirs(tfrecord_path) exception', e)

    qm9 = load_qm9()
    rxns = load_rxns()
    pdbs = load_pdbs()
    write_shards(qm9, rxns, pdbs)


if __name__ == '__main__':
    main()

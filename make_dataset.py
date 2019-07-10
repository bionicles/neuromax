from progiter import ProgIter
import tensorflow as tf
from pymol import cmd
import numpy as np
import random
import shutil
import time
import csv
import os
CSV_FILE_NAME = 'sorted-less-than-256.csv'
SHARDS_PER_DATASET = 1
ITEMS_PER_SHARD = 4
DTYPE = tf.float32
MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE = 1, 64
P_UNDOCK = 0.5
P_UNFOLD = 0.5

# quantum -> datasets/xyz: https://ndownloader.figshare.com/files/3195389
# chem -> datasets/rxn: ftp://ftp.expasy.org/databases/rhea/ctfiles/rhea-rxn.tar.gz
# chem -> datasets/mol: ftp://ftp.expasy.org/databases/rhea/ctfiles/rhea-mol.tar.gz
# cif datasets use datasets/csv lists of RCSB cif ids

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


def load_folder(type):
    return os.listdir(os.path.join('.', 'datasets', type))


def load_proteins():
    with open(os.path.join('.', 'datasets', 'csv', CSV_FILE_NAME)) as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        return [item.strip() for item in rows[0]]


def undock(chains, type):
    for chainOrName in chains:
        selection_string = f'chain {chainOrName}' if type is 'cif' else chainOrName
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
        pass
        # print('failed to set dihedral at ', name, index)
        # print(e)


def get_positions(model):
    positions = np.array(model.get_coord_list())
    return tf.convert_to_tensor(positions, dtype=DTYPE)


def get_features(model):
    features = np.array([get_atomic_features(atom) for atom in model.atom if atom.symbol != "gdb"])
    return tf.convert_to_tensor(features, dtype=DTYPE)


def get_masses(model):
    masses = np.array([[atom.get_mass()] for atom in model.atom if atom.symbol != "gdb"])
    return tf.convert_to_tensor(masses, dtype=DTYPE)


def get_numbers(model):
    numbers = np.array([[elements[atom.symbol.lower()]] for atom in model.atom if atom.symbol != "gdb"])
    return tf.convert_to_tensor(numbers, dtype=DTYPE)


def get_atomic_features(atom):
    return np.array([atom.formal_charge,
                     atom.vdw,
                     atom.b,
                     atom.q,
                     atom.get_free_valence(0)], dtype=np.float32)


def clean_xyz(path):
    print(f"clean xyz at {path}")
    try:
        os.remove("./datasets/tmp.xyz")
    except Exception as e:
        print('couldnt delete tmp.xyz')
    with open(path, "r") as start_file:
        with open("./datasets/tmp.xyz", "a") as out_file:
            for line in start_file.readlines()[1:]:
                splits = line.split("\t")
                if "gdb" in splits[0]:
                    quantum_target = tf.convert_to_tensor([float(element) for element in splits[1:-1]])
                elif "gdb" not in splits[0] and splits[0].isalpha():
                    out_file.write('\t'.join(splits[0:4]) + '\n')
                else:
                    break
            return quantum_target


def get_reactants_products(path):
    print(f"getting reactants and products for {path}")
    line_is_n_reactants_products = False
    line_is_molfile = False
    mol_data = open(path)
    reactants = []
    products = []
    molfiles = []
    for line in mol_data:
        if line_is_n_reactants_products:
            reactants_products = line.strip().split(" ")
            n_reactants = int(reactants_products[0])
            n_products = int(reactants_products[-1])
            line_is_n_reactants_products = False
        elif line_is_molfile and ":" in line:
            molfile = line.replace(':', '_')
            molfile = molfile.replace('\n', '')
            molfiles.append(f'{molfile}.mol')
            line_is_molfile = False
        if "RHEA:release" in line:
            line_is_n_reactants_products = True
        if "$MOL" in line:
            line_is_molfile = True
    for molfile in molfiles:
        if n_reactants > 0:
            reactants.append(molfile)
            n_reactants -= 1
        elif n_products > 0:
            products.append(molfile)
            n_products -= 1
    return reactants, products


def clean_pymol():
    cmd.remove("all and not (alt '')")  # remove alternate conformations
    cmd.alter("all", 'alt=""')
    cmd.remove('solvent')


def load(type, id):
    # we clear pymol, make filename and path
    quantum_target = []
    target_features = []
    cmd.delete('all')
    file_name = f'{id}.{type}' if type is 'cif' else id
    dataset_path = os.path.join('.', 'datasets', type)
    path = os.path.join(dataset_path, file_name)
    # we load the target
    print(f'load {id} {path}')
    if type is "cif":
        if not os.path.exists(path):
            cmd.fetch(id, path=dataset_path)
        elif os.path.exists(path):
            cmd.load(path)
    elif type is "xyz":
        quantum_target = clean_xyz(path)
        cmd.load(os.path.join(".", "datasets", "tmp.xyz"))
    elif type is "rxn":
        reactants, products = get_reactants_products(path)
        print(f"{str(reactants)} ---> {str(products)}")
        for product in products:
            product_path = os.path.join('.', 'datasets', 'mol', product)
            cmd.load(product_path)
    # make target
    clean_pymol()
    model = cmd.get_model('all', 1)
    target_positions = get_positions(model)
    if type is not "xyz":
        quantum_target = tf.zeros((15), dtype=tf.float32)
    if type is "rxn":
        target_features = get_features(model)
        target_masses = get_masses(model)
        target_numbers = get_numbers(model)
        target_features = tf.concat([target_features, target_masses, target_numbers], -1)
        cmd.delete('all')
        for reactant in reactants:
            reactant_path = os.path.join('.', 'datasets', 'mol', reactant)
            cmd.load(reactant_path)
        clean_pymol()
        model = cmd.get_model('all', 1)
    # make the model inputs
    if type is 'cif':
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
    masses = get_masses(model)
    numbers = get_numbers(model)
    features = tf.concat([features, masses, numbers], -1)
    if type is not "rxn":
        target_features = features
    return make_example(type, id, target_positions, positions, features, masses, quantum_target, target_features)


def make_example(type, id, target_positions, positions, features, masses, quantum_target, target_features):
    [print(x.shape) for x in [target_positions, positions, features, masses, quantum_target, target_features]]
    example = tf.train.SequenceExample()
    # non-sequential features
    example.context.feature["type"].bytes_list.value.append(bytes(type, 'utf-8'))
    example.context.feature["id"].bytes_list.value.append(bytes(id, 'utf-8'))
    example.context.feature["quantum_target"].bytes_list.value.append(tf.io.serialize_tensor(quantum_target).numpy())
    # sequential features
    fl_target_positions = example.feature_lists.feature_list["target_positions"]
    fl_target_positions.feature.add().bytes_list.value.append(tf.io.serialize_tensor(target_positions).numpy())
    fl_target_features = example.feature_lists.feature_list["target_features"]
    fl_target_features.feature.add().bytes_list.value.append(tf.io.serialize_tensor(target_features).numpy())
    fl_positions = example.feature_lists.feature_list["positions"]
    fl_positions.feature.add().bytes_list.value.append(tf.io.serialize_tensor(positions).numpy())
    fl_features = example.feature_lists.feature_list["features"]
    fl_features.feature.add().bytes_list.value.append(tf.io.serialize_tensor(features).numpy())
    fl_masses = example.feature_lists.feature_list["masses"]
    fl_masses.feature.add().bytes_list.value.append(tf.io.serialize_tensor(masses).numpy())
    # serialized
    return example.SerializeToString()


def write_shards():
    problems = []
    qm9 = load_folder("xyz")
    rxns = load_folder("rxn")
    proteins = load_proteins()
    for dataset, type in [(qm9, "xyz"), (rxns, "rxn"), (proteins, "cif")]:
        shard_number, item_number = 0, 0
        for dataset_item in ProgIter(dataset, verbose=1):
            if item_number % ITEMS_PER_SHARD is 0:
                try:
                    writer.close()
                except Exception as e:
                    print('writer.close() exception', e)
                shard_path = os.path.join('.', 'datasets', 'tfrecord', type)
                if not os.path.exists(shard_path):
                    os.makedirs(shard_path)
                shard_path = os.path.join(shard_path, str(shard_number) + '.tfrecord')
                writer = tf.io.TFRecordWriter(shard_path, 'ZLIB')
                shard_number += 1
                if shard_number > SHARDS_PER_DATASET:
                    break
            try:
                data = load(type, dataset_item.lower())
                writer.write(data)
                print('wrote', dataset_item, 'to', shard_path)
            except Exception as e:
                print('failed on', shard_number, dataset_item, shard_path)
                print(e)
                problems.append([shard_number, dataset_item, e])
            item_number += 1
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

    write_shards()


if __name__ == '__main__':
    main()

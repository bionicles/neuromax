from progiter import ProgIter
import tensorflow as tf
from pymol import cmd, util
import numpy as np
import random
import shutil
import csv
import os
DATASETS_ROOT = os.path.join('.', 'src', 'nurture', 'mol', 'datasets')
CSV_FILE_NAME = 'sorted-less-than-256.csv'
TEMP_PATH = "archive/temp"
SHARDS_PER_DATASET = 100000
ITEMS_PER_SHARD = 16
DELETE_RECORDS = False
DTYPE = tf.float32
MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE = -64, 64
P_UNDOCK = 1
P_UNFOLD = 0.5
MAX_ATOMS = 6000

tfrecord_path = os.path.join(DATASETS_ROOT, 'tfrecord')
if DELETE_RECORDS:
    shutil.rmtree(os.path.join(tfrecord_path))
try:
    os.mkdir(tfrecord_path)
    os.mkdir(os.path.join(tfrecord_path, 'cif'))
except Exception as e:
    print('os.path.makedirs(tfrecord_path) exception', e)
NUM_FILES = len(os.listdir(os.path.join(tfrecord_path, 'cif')))
PREVIOUS_PROTEINS = ITEMS_PER_SHARD * NUM_FILES


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


# def load_folder(type):
#     return os.listdir(os.path.join('.', 'datasets', type))


def take_screenshot(step):
    screenshot_path = f"{TEMP_PATH}/{step}.png"
    print(f"SCREENSHOT!, {screenshot_path}")
    cmd.zoom("all")
    cmd.png(screenshot_path, width=256, height=256)


def prepare_pymol():
    cmd.clip("near", 100000)
    cmd.clip("slab", 100000)
    cmd.clip("far", -100000)
    cmd.show("spheres")
    cmd.unpick()
    util.cbc()


def load_proteins(file_name):
    csv_path = os.path.join(DATASETS_ROOT, 'csv', file_name)
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        return [item.strip() for item in rows[0][PREVIOUS_PROTEINS:]]


def undock(chainsOrNames, type):
    print('undocking', chainsOrNames, 'type', type)
    for chainOrName in chainsOrNames:
        if type == 'cif':
            selection_string = f'chain {chainOrName}'
        else:
            selection_string = chainOrName
        translation_vector = [
            random.randrange(MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE),
            random.randrange(MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE),
            random.randrange(MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE)]
        print('translate', selection_string, translation_vector)
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
    print(selection_string_array)
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


def clean_pymol():
    cmd.remove("all and not (alt '')")  # remove alternate conformations
    cmd.alter("all", 'alt=""')
    cmd.remove('solvent')


def load(type, id, screenshot=False):
    # we clear pymol, make filename and path
    cmd.delete('all')
    file_name = f'{id}.{type}' if type is 'cif' else id
    dataset_path = os.path.join(DATASETS_ROOT, type)
    path = os.path.join(dataset_path, file_name)
    # we load the target
    print(f'load {id} {path}')
    if type is "cif":
        if not os.path.exists(path):
            cmd.fetch(id, path=dataset_path)
        elif os.path.exists(path):
            cmd.load(path)
    # make target
    clean_pymol()
    model = cmd.get_model('all', 1)
    target = get_positions(model)
    # make the model inputs
    if screenshot:
        prepare_pymol()
        take_screenshot("0")
    chains = cmd.get_chains('all')
    if len(chains) == 0:
        return False
    if random.random() < P_UNDOCK:
        print('undocking')
        undock(chains, type)
    if random.random() < P_UNFOLD:
        print('unfolding')
        unfold(chains)
    model = cmd.get_model('all', 1)
    positions = get_positions(model)
    if positions.shape[0] > MAX_ATOMS:
        print(f"{positions.shape[0]} IS TOO MANY ATOMS")
        return False
    features = get_features(model)
    masses = get_masses(model)
    numbers = get_numbers(model)
    features = tf.concat([features, masses, numbers], -1)
    target = tf.concat([target, features], -1)
    return make_example(id, target, positions, features, masses)


def make_example(id, target, positions, features, masses):
    example = tf.train.SequenceExample()
    # non-sequential features
    example.context.feature["id"].bytes_list.value.append(bytes(id, 'utf-8'))
    # sequential features
    fl_target = example.feature_lists.feature_list["target"]
    fl_target.feature.add().bytes_list.value.append(tf.io.serialize_tensor(target).numpy())
    fl_positions = example.feature_lists.feature_list["positions"]
    fl_positions.feature.add().bytes_list.value.append(tf.io.serialize_tensor(positions).numpy())
    fl_features = example.feature_lists.feature_list["features"]
    fl_features.feature.add().bytes_list.value.append(tf.io.serialize_tensor(features).numpy())
    fl_masses = example.feature_lists.feature_list["masses"]
    fl_masses.feature.add().bytes_list.value.append(tf.io.serialize_tensor(masses).numpy())
    # serialized
    return example.SerializeToString()


def write_shards(tfrecord_path):
    global PREVIOUS_PROTEINS
    problems = []
    proteins = load_proteins(CSV_FILE_NAME)
    for dataset, type in [(proteins, "cif")]:
        shard_number = NUM_FILES-1 if NUM_FILES > 0 else 0
        for dataset_item in ProgIter(dataset, verbose=1):
            if PREVIOUS_PROTEINS % ITEMS_PER_SHARD is 0:
                try:
                    writer.close()
                except Exception as e:
                    print('writer.close() exception', e)
                shard_path = os.path.join(tfrecord_path, type)
                if not os.path.exists(shard_path):
                    os.makedirs(shard_path)
                shard_path = os.path.join(shard_path, str(shard_number) + '.tfrecord')
                writer = tf.io.TFRecordWriter(shard_path, 'ZLIB')
                shard_number += 1
                if shard_number > SHARDS_PER_DATASET:
                    break
            try:
                data = load(type, dataset_item.lower())
                if data:
                    writer.write(data)
                    print('wrote', dataset_item, 'to', shard_path)
                    PREVIOUS_PROTEINS += 1
                else:
                    print('skipped writing', dataset_item, 'to', shard_path)
            except Exception as e:
                print('failed on', shard_number, dataset_item, shard_path)
                print(e)
                problems.append([shard_number, dataset_item, e])
    print('problem children:')
    [print(problem) for problem in problems]
    print('done!')


def main():
    write_shards(tfrecord_path)


if __name__ == '__main__':
    main()

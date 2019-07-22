from scipy.optimize import linear_sum_assignment
import tensorflow as tf
import os

B, L, K = tf.keras.backend, tf.keras.layers, tf.keras


@tf.function
def parse_item(example):
    context_features = {  # 'quantum_target': tf.io.FixedLenFeature([], dtype=tf.string),
                        'type': tf.io.FixedLenFeature([], dtype=tf.string),
                        'id': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = {'target_positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'target_numbers': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'features': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'numbers': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'masses': tf.io.FixedLenSequenceFeature([], dtype=tf.string)}
    context, sequence = tf.io.parse_single_sequence_example(example, context_features=context_features, sequence_features=sequence_features)
    target_positions = tf.reshape(tf.io.parse_tensor(sequence['target_positions'][0], tf.float32), [-1, 3])
    target_numbers = tf.reshape(tf.io.parse_tensor(sequence['target_numbers'][0], tf.float32), [-1, 1])
    positions = tf.reshape(tf.io.parse_tensor(sequence['positions'][0], tf.float32), [-1, 3])
    features = tf.reshape(tf.io.parse_tensor(sequence['features'][0], tf.float32), [-1, 7])
    numbers = tf.reshape(tf.io.parse_tensor(sequence['numbers'][0], tf.float32), [-1, 1])
    masses = tf.reshape(tf.io.parse_tensor(sequence['masses'][0], tf.float32), [-1, 1])
    # quantum_target = tf.io.parse_tensor(context['quantum_target'], tf.float32)
    masses = tf.concat([masses, masses, masses], 1)
    n_source_atoms = tf.shape(positions)[0]
    n_target_atoms = tf.shape(target_positions)[0]
    type_string = context['type']
    id_string = context['id']
    return type_string, id_string, n_source_atoms, n_target_atoms, target_positions, positions, features, masses, target_numbers, numbers


def read_shards(datatype):
    print("read_shards", datatype)
    dataset_path = os.path.join('.', 'datasets', 'tfrecord', datatype)
    n_records = len(os.listdir(dataset_path))
    filenames = [os.path.join(dataset_path, str(i) + '.tfrecord') for i in range(n_records)]
    dataset = tf.data.TFRecordDataset(filenames, 'ZLIB')
    dataset = dataset.map(map_func=parse_item, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(1)
    return n_records, dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

#
# def move_atom(xyz):
#     global atom_index
#     xyz = xyz.tolist()
#     cmd.translate(xyz, f'id {str(atom_index)}')
#     atom_index += 1
#
#
# def move_atoms(velocities):
#     global atom_index
#     atom_index = 0
#     np.array([move_atom(xyz) for xyz in velocities])
#
#
# def prepare_pymol():
#     cmd.show(GIF_STYLE)
#     cmd.unpick()
#     util.cbc()
#
#
# def make_gif(pdb_name, trial_name, pngs_path):
#     gif_name = f'{pdb_name}-{trial_name}.gif'
#     gif_path = os.path.join(".", "gifs", gif_name)
#     imagepaths, images = [], []
#     for stackname in os.listdir(pngs_path):
#         print('processing', stackname)
#         filepath = os.path.join(pngs_path, stackname)
#         imagepaths.append(filepath)
#     imagepaths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
#     for imagepath in imagepaths:
#         image = imageio.imread(imagepath)
#         images.append(image)
#     print('saving gif to', gif_path)
#     imageio.mimsave(gif_path + pdb_name + ".gif", images)


# @tf.function
def igsp(source_positions, source_numbers, target_positions, target_numbers):
    print("tracing igsp")
    source_positions, source_numbers, target_positions, target_numbers = [tf.squeeze(x, 0) for x in [source_positions, source_numbers, target_positions, target_numbers]]
    columns, rows = tf.range(tf.shape(target_positions)[0]), tf.range(tf.shape(target_positions)[0])
    source_centroid = tf.reduce_mean(source_positions, axis=0)
    target_centroid = tf.reduce_mean(target_positions, axis=0)
    source_positions_copy = tf.identity(source_positions) - source_centroid
    target_positions_copy = tf.identity(target_positions) - target_centroid
    translation = target_centroid - source_centroid
    feature_distances = get_distances(source_numbers, target_numbers)
    rotation = tf.ones((3,3))
    for igsp_step in range(2):
        euclidean_distances = get_distances(source_positions_copy, target_positions_copy)
        compound_distances = euclidean_distances + 10. * feature_distances
        rows, columns = tf.py_function(linear_sum_assignment, [compound_distances], [tf.int32, tf.int32])
        rows = tf.cast(rows, tf.int32)
        columns = tf.cast(columns, tf.int32)
        ordered_source_positions = tf.gather(source_positions_copy, columns)
        ordered_target_positions = tf.gather(target_positions_copy, rows)
        covariance = B.dot(tf.transpose(ordered_source_positions), ordered_target_positions)
        s, u, v = tf.linalg.svd(covariance)
        d = tf.linalg.det(v * tf.transpose(u))
        temporary_rotation = v * tf.linalg.diag([1., 1., d]) * tf.transpose(u)
        source_positions_copy = tf.tensordot(source_positions_copy, temporary_rotation, 1)
        rotation = temporary_rotation * rotation
    return rows, columns, translation, rotation


def get_work_loss(n_source_atoms, n_target_atoms, target_positions, target_numbers, positions, numbers, masses, velocities):
    print("tracing get_work_loss")
    rows, columns, translation, rotation = igsp(positions, numbers, target_positions, target_numbers)
    aligned = tf.linalg.matmul(positions, rotation)
    aligned = positions + translation
    gathered_positions = tf.squeeze(tf.gather(aligned, columns, axis=1), 0)
    gathered_target = tf.squeeze(tf.gather(target_positions, rows, axis=1), 0)
    distances = tf.linalg.diag_part(get_distances(gathered_positions, gathered_target))
    gathered_masses, _, _ = tf.split(tf.squeeze(tf.gather(masses, columns, axis=1), 0), 3, axis=-1)
    return tf.multiply(tf.squeeze(gathered_masses, -1), (2. * distances))


def get_pdb_loss(target_positions, positions, masses):
    masses, _, _ = tf.split(tf.squeeze(masses, 0), 3, axis=-1)
    return tf.multiply(tf.squeeze(masses, -1), 2. * tf.linalg.diag_part(get_distances(positions, target_positions)))


@tf.function
def get_distances(a, b):  # L2
    print("tracing get_distances")
    return B.sqrt(B.sum(B.square(tf.expand_dims(a, 0) - tf.expand_dims(b, 1)), axis=-1))


# @tf.function
# def get_distances(a, b):  # L1
#     print("tracing get_distances")
#     return B.sum(B.abs(tf.expand_dims(a, 0) - tf.expand_dims(b, 1)), axis=-1))

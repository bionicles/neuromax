import tensorflow as tf
import ot
B = tf.keras.backend
D_SPACE = 3
D_FEATURES = 7
DTYPE = tf.float32

@tf.function
def parse_item(example):
    context_features = {'quantum_target': tf.io.FixedLenFeature([], dtype=tf.string),
                        'type': tf.io.FixedLenFeature([], dtype=tf.string),
                        'id': tf.io.FixedLenFeature([], dtype=tf.string)}
    sequence_features = {'target_positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'target_features': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'features': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
                         'masses': tf.io.FixedLenSequenceFeature([], dtype=tf.string)}
    context, sequence = tf.io.parse_single_sequence_example(example, context_features=context_features, sequence_features=sequence_features)
    target_positions = tf.reshape(tf.io.parse_tensor(sequence['target_positions'][0], tf.float32), [-1, 3])
    target_features = tf.reshape(tf.io.parse_tensor(sequence['target_features'][0], tf.float32), [-1, 7])
    target_numbers = tf.reshape(tf.io.parse_tensor(sequence['target_numbers'][0], tf.float32), [-1, 1])
    positions = tf.reshape(tf.io.parse_tensor(sequence['positions'][0], tf.float32), [-1, 3])
    features = tf.reshape(tf.io.parse_tensor(sequence['features'][0], tf.float32), [-1, 7])
    masses = tf.reshape(tf.io.parse_tensor(sequence['masses'][0], tf.float32), [-1, 1])
    numbers = tf.reshape(tf.io.parse_tensor(sequence['numbers'][0], tf.float32), [-1, 1])
    quantum_target = tf.io.parse_tensor(context['quantum_target'], tf.float32)
    masses = tf.concat([masses, masses, masses], 1)
    n_atoms = tf.shape(positions)[0]
    type = context['type']
    id = context['id']
    target_features = features
    return (type, id, n_atoms, target_positions, positions, features, masses, quantum_target, target_features, target_numbers, numbers)


def read_shards(datatype):
    print("read_shards", datatype)
    dataset_path = os.path.join('.', 'datasets', 'tfrecord', datatype)
    n_records = len(os.listdir(dataset_path))
    filenames = [os.path.join(dataset_path, str(i) + '.tfrecord') for i in range(n_records)]
    dataset = tf.data.TFRecordDataset(filenames, 'ZLIB')
    dataset = dataset.map(map_func=parse_item, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(1)
    return n_records, dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)



@tf.function(input_signature=[tf.TensorSpec(shape=(None, D_SPACE), dtype=DTYPE)])
def get_distances(a, b):  # target is treated like current
    return tf.reduce_sum(tf.math.abs(tf.expand_dims(a, 0) - tf.expand_dims(b, 1)), axis=-1)  # N, N, 1



@tf.function
def show(m):
    tf.print(tf.shape(m))



def cos(A, B):
    Aflat = tf.keras.backend.flatten(A)
    Bflat = tf.keras.backend.flatten(B)
    return (tf.math.dot( Aflat, Bflat ) /
            tf.math.maximum( tf.norm(Aflat) * tf.norm(Bflat), 1e-10 ))


def igsp(source_positions, source_numbers, target_positions, target_numbers):
    translation = tf.reduce_mean(target_positions, axis=0) - tf.reduce_mean(source_positions, axis=0)
    source_positions_copy = tf.identity(source_positions)
    n_source_atoms = tf.shape(source_positions)[0]
    change_in_rotation =  1000
    igsp_step = 0
    while change_in_rotation > (10 * 3.14159/180) and igsp_step < 20:
        euclidean_distance = get_distances(source_positions_copy, target_positions)
        feature_distance = 1000 * get_distances(source_numbers, target_numbers)
        feature_weight = tf.math.exp(-igsp_step / n_source_atoms)
        euclidean_weight = 1 - feature_weight + 0.001
        compound_distance = euclidean_weight * euclidean_distance + feature_weight * feature_distance
        rows, columns = scipy.optimize.linear_sum_assignment(compound_distance)
        ordered_source_positions = tf.gather(source_positions, columns) - tf.reduce_mean(ordered_source_positions, axis=0)
        ordered_target_positions = tf.gather(target_positions, rows) - tf.reduce_mean(ordered_target_positions, axis=0)
        covariance = tf.dot(ordered_source_positions, tf.transpose(ordered_target_positions))
        s, u, v = tf.linalg.svd(covariance)
        d = tf.det(v * tf.transpose(u))
        temporary_rotation = v * tf.linalg.diag([1,1,d]) * tf.transpose(u)
        source_positions_copy = temporary_rotation * source_positions
        change_in_rotation = cos(rotation, temporary_rotation)
        igsp_step += 1
    return rows, columns, translation, rotation

def get_loss(target_positions, positions, velocities, masses):


# @tf.function
def test_noobstein(N, M):
    reactions = read_shards('rxn')
    # align the current and target with icp
    # get distances between source and target atoms
    # loss = mass * 2 * distance - velocity

n = 1000
while True:
    test_noobstein(n, n)
    n = n + 100

import tensorflow as tf
import ot

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
def get_distances(xyz):  # target is treated like current
    return tf.reduce_sum(tf.math.abs(tf.expand_dims(xyz, 0) - tf.expand_dims(xyz, 1)), axis=-1)  # N, N, 1



@tf.function
def show(m):
    tf.print(tf.shape(m))



def quaternion_mult(q,r):
    return [r[0]q[0]-r[1]q[1]-r[2]q[2]-r[3]q[3],
            r[0]q[1]+r[1]q[0]-r[2]q[3]+r[3]q[2],
            r[0]q[2]+r[1]q[3]+r[2]q[0]-r[3]q[1],
            r[0]q[3]-r[1]q[2]+r[2]q[1]+r[3]q[0]]

def point_rotation_by_quaternion(point,q):
    r = [0]+point
    q_conj = [q[0],-1q[1],-1q[2],-1*q[3]]
    return quaternion_mult(quaternion_mult(q,r),q_conj)[1:]

def icp(source_positions, source_numbers, target_positions, target_numbers):

    # Initialization
    Rt = tf.linalg.identity() # identity matrix?
    Delta_r =  1000
    delta_t = 1000
    k = 0
    while translation > translation_cutoff and rotation > rotation_cutoff:
        euclidean_distance = get_distances(source_positions, target_positions)
        W_fd = tf.math.exp(-k / m)  # number of atoms of current
        W_ed = 1 - W_fd
        compound_distance = euclidean_weight*euclidean_distance + feature_weight * feature_distance
        if k == 0:
            Tcd = mean(compound_distance) + p[0] segma_CD
        else:
            Tcd = p[1]*euclidean_weight * mean(ex_euclidean_distance) + p[2]*W_fd*mean(ex_feature_distance) # p threshold param
        e(p, q)  # WTF is that, 10 step in the algorithm
        M = scipy.optimize.linear_sum_assignment(Tcd)  # set of correspondences between target and current
        Rtemp = SVD(M)
        current_position = Rtemp*current_position
        delta_r and delta_t from Rtemp


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

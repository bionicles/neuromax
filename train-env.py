# train-env.py - why?: train faster with tfrecords not pymol
import tensorflow as tf
import gym

K = tf.keras

pluck = lambda dict, *args: (dict[arg] for arg in args)

class TrainEnv():
    def __init__(self, n_calls, batch_size):
        self.config = env_config
        self.data = self.read_shards()
        example = self.reset()
        self.observation_space = gym.spaces.Box()

    def loss(self, positions, velocities, masses):
        # position
        position_error = K.losses.mean_squared_error(self.initial_positions, positions)
        position_error *= position_error_weight
        # shape
        distances = self.calculate_distances(positions)
        shape_error = K.losses.mean_squared_error(self.initial_distances, distances)
        shape_error *= shape_error_weight
        # action energy
        kinetic_energy = (masses / 2) * (velocities ** 2)
        potential_energy = force_field
        action *= action_weight
        return position_error + shape_error + action

    def reset(self):
        state = self.data.take(1)
        initial_positions, positions = pluck(state, 'initial_positions', 'positions')
        initial_distances = pluck(state, 'initial_distances')
        force_field = pluck(state, 'force_field')
        stop_loss = self.loss(state) * self.config.stop_loss()
        return tf.concat([position, velocity, features])

    def step(self, force_field, info):
        positions, velocities = self.move_atoms(force_field)
        reward = -1 * (position_error + shape_error + kinetic_energy)
        observation = tf.concat([positions, velocities], 2)
        done = loss > stop_loss
        return observation, reward, done, info

    def move_atoms(self, force_field):
        acceleration = force_field / masses
        noise = random_normal((num_atoms, 3), 0, NOISE, dtype='float16')
        velocities += acceleration + noise
        positions += velocities
        observation =
        return observation, reward, done, info


    # BEGIN READING HELPERS
    def read_shards():
        path = os.path.join('.', 'tfrecords')
        recordpaths = []
        for name in os.listdir(path):
            recordpaths.append(os.path.join(path, name))
        dataset = tf.data.TFRecordDataset(recordpaths, 'ZLIB')
        dataset.map(map_func=self.parse_example,
                    num_parallel_calls=self.config.num_parallel_calls)
        dataset = dataset.batch(batch_size=self.config.batch_size)
        dataset = dataset.prefetch(buffer_size=self.config.buffer_size)
        return dataset, recordpaths


    def parse_example(self, example):
        context_features={
            'num_atoms': tf.io.FixedLenFeature([], dtype=tf.int64),  # scalar
            'protein': tf.io.FixedLenFeature([], dtype=tf.string),  # scalar
        }
        sequence_features={
            'initial_positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),  # num_atoms * 3
            'positions': tf.io.FixedLenSequenceFeature([], dtype=tf.string),  # num_atoms * 3
            'features': tf.io.FixedLenSequenceFeature([], dtype=tf.string),  # num_atoms * 16
            'masses': tf.io.FixedLenSequenceFeature([], dtype=tf.string)  # num_atoms * 3
        }
        context, sequence = tf.io.parse_single_sequence_example(example, context_features=context_features, sequence_features=sequence_features)
        initial_positions = parse_feature(sequence['initial_positions'][0], 3)
        initial_distances = self.calculate_distances(initial_positions)
        num_atoms = context['num_atoms'].numpy()
        features = parse_feature(sequence['features'][0], 9)
        masses = parse_feature(sequence['masses'][0], 1)
        features = tf.concat([masses, features], 2)
        masses = tf.concat([masses, masses, masses], 2)
        return {
            'initial_distances': self.calculate_distances(initial_positions),
            'positions': parse_feature(sequence['positions'][0], 3),
            'protein': context['protein'].numpy().decode(),
            'initial_positions': initial_positions,
            'num_atoms_squared': num_atoms ** 2,
            'features': features,
            'num_atoms': num_atoms,
            'masses': masses,
            }


def parse_feature(string, d):
    tensor = tf.io.parse_tensor(string, tf.int8)
    return tf.reshape(tensor, [-1, d])



def calculate_distances(positions):
    distances = tf.reduce_sum(positions * positions, 1)
    distances = tf.reshape(distances, [-1, 1])
    return distances - 2 * tf.matmul(positions, tf.transpose(positions)) + tf.transpose(distances)


def show_parsed_example(parsed):
    print('')
    [print(k, t.numpy()[0]) if isinstance(t, tf.Tensor)
     else print(k, t) for k, t in parsed.items()]
    # END READING HELPERS


if __name__ == "__main__":
    env = TrainEnv(env_config)
    for i in range(10):
        print(env.reset())

# test-env - why?: test TRAINED models on GIFs
import tensorflow as tf
import gym

# globals
# experiment, episode, step, num_atoms, num_atoms_squared = 0, 0, 0, 0, 0
# velocities, masses, chains, model = [], [], [], []
# episode_images_path, episode_stacks_path = '', ''
# episode_path, run_path = '', ''
# ROOT = os.path.abspath('.')
# SAVE_MODEL = False
# atom_index = 0
# pdb_name = ''
# TIME = ''
# task
# IMAGE_SIZE = 256
# POSITION_VELOCITY_LOSS_WEIGHT, SHAPE_LOSS_WEIGHT = 10, 1
# MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE = 8, 32
# SCREENSHOT_EVERY = 5
# NOISE = 0.002
# WARMUP = 1000
# BUFFER = 42
# BIGGEST_FIRST_IF_NEG = 1
# RANDOM_PROTEINS = False

class MolEnv(gym.Env):
    def __init__(self, records):
        print('init MolEnv')
        self.dataset = tf.data.TFRecordDataSet(records)


    def _parse_function(proto):
        # define your tfrecord again. Remember that you saved your image as a string.
            keys_to_features = {
        'initial': float_value(initial),  # num_atoms * 6
        'num_atoms_squared': tf.FixedLenFeature(tf.uint32),  # scalar
        'num_atoms': tf.FixedLenFeature(tf.uint32),  # scalar
        'initial_distances': float_value(initial_distances), # num_atoms * num_atoms
        'features': float_value(features),  # num_atoms * 16
        'positions': float_value(positions),  # num_atoms * 3
        'velocities': float_value(velocities),  # num_atoms * 3
        'current': float_value(current),  # num_atoms * 6
        'masses': float_value(masses),  # num_atoms * 3
    }

        # Load one example
        parsed_features = tf.parse_single_example(proto, keys_to_features)

        # Turn your saved image string into an array
        parsed_features['image'] = tf.decode_raw(
            parsed_features['image'], tf.uint8)

        return parsed_features['image'], parsed_features["label"]


    def create_dataset(filepath):
        dataset = tf.data.TFRecordDataset(filepath)
        dataset = dataset.map(_parse_function, num_parallel_calls=8)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(SHUFFLE_BUFFER)
        dataset = dataset.batch(BATCH_SIZE)
        iterator = dataset.make_one_shot_iterator()
        image, label = iterator.get_next()
        image = tf.reshape(image, [-1, 256, 256, 1])
        label = tf.one_hot(label, NUM_CLASSES)

    def reset(self):
        if self.mode == 'training':
            return self.dataset.take(self.batch_size)
        else:
            return self.load_protein()

    def step(self, force_field):
        acceleration = force_field / masses
        noise = random_normal((num_atoms, 3), 0, NOISE, dtype='float32')
        velocities += acceleration + noise
        if screenshot:
            iter_velocity = tf.squeeze(velocities)
            atom_index = 0
            np.array([move_atom(xyz) for xyz in iter_velocity])
        positions = tf.math.add(positions, velocities)
        return observation, reward, done, info

    def loss(action, initial):
        position, velocities = move_atoms(action)
        # meta distance (shape loss)
        current_distances = calculate_distances(position)
        shape_loss = tf.losses.mean_squared_error(current_distances, initial_distances)
        shape_loss /= num_atoms_squared
        shape_loss *= SHAPE_LOSS_WEIGHT
        # normal distance (position + velocity loss)
        current = tf.concat([positions, velocities], 1)
        pv_loss = tf.losses.mean_squared_error(current, initial)
        pv_loss /= num_atoms
        pv_loss *= POSITION_VELOCITY_LOSS_WEIGHT
        # loss value (sum)
        loss_value = shape_loss + pv_loss
        if VERBOSE:
            print('shape', round(shape_loss.numpy().tolist(), 2),
                  'pv', round(pv_loss.numpy().tolist(), 2),
                  'total loss', round(loss_value.numpy().tolist(), 2))
        return loss_value

    def calculate_distances(position_tensor):
        distances = tf.reduce_sum(position_tensor * position_tensor, 1)
        distances = tf.reshape(distances, [-1, 1])
        distances = distances - 2 * tf.matmul(
            position_tensor,
            tf.transpose(position_tensor)) + tf.transpose(distances)
        return distances

    def move_atom(xyz):
        atom_selection_string = 'id ' + str(atom_index)
        xyz = xyz.numpy().tolist()
        cmd.translate(xyz, atom_selection_string)
        atom_index += 1

    def load_pedagogy():
        with open(os.path.join('.', PEDAGOGY_FILE_NAME)) as csvfile:
            reader = csv.reader(csvfile)
            results = []
            for row in reader:
                for item in row:
                    item = item.strip()
                    results.append(item)
            pedagogy = results
        print('loaded', len(pedagogy), 'structures')


    def prepare_pymol():
        cmd.remove('solvent')
        util.cbc()
        cmd.set('depth_cue', 0)
        cmd.show('spheres')
        cmd.zoom('all', buffer=BUFFER, state=-1)
        cmd.center('all')
        cmd.bg_color('black')

    def make_gif():
        gif_name = '{}-{}-{}.gif'.format(episode, pdb_name, TIME)
        gif_path = os.path.join(episode_path, gif_name)
        print('episode path', episode_path)
        imagepaths, images = [], []
        for stackname in os.listdir(episode_stacks_path):
            print('processing', stackname)
            filepath = os.path.join(episode_stacks_path, stackname)
            imagepaths.append(filepath)
        imagepaths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for imagepath in imagepaths:
            image = imageio.imread(imagepath)
            images.append(image)
        print('saving gif to', gif_path)
        imageio.mimsave(gif_path, images)
        shutil.rmtree(episode_images_path)

    def make_image():
        step_string = str(step)
        png_paths = take_pictures(step_string)
        images = [np.asarray(Image.open(png)) for png in png_paths]
        vstack = np.vstack(images)
        image_path = os.path.join(episode_stacks_path, step_string + '.png')
        vstack_img = Image.fromarray(vstack)
        x, y = vstack_img.size
        draw = ImageDraw.Draw(vstack_img)
        model_id_string = 'Bit Pharma NEUROMAX: ' + TIME
        x3, y3 = draw.textsize(model_id_string)
        draw.text(((x-x3)/2, y-42), model_id_string, (255, 255, 255))
        print('screenshot', image_path)
        vstack_img.save(image_path)

    def take_pictures(step_string):
        cmd.deselect()
        png_path_x = os.path.join(episode_pngs_path, step_string + '-X.png')
        cmd.png(png_path_x, width=IMAGE_SIZE, height=IMAGE_SIZE)
        time.sleep(0.02)
        cmd.rotate('y', 90)
        png_path_y = os.path.join(episode_pngs_path, step_string + '-Y.png')
        cmd.png(png_path_y, width=IMAGE_SIZE, height=IMAGE_SIZE)
        time.sleep(0.02)
        cmd.rotate('y', -90)
        cmd.rotate('z', 90)
        png_path_z = os.path.join(episode_pngs_path, step_string + '-Z.png')
        cmd.png(png_path_z, width=IMAGE_SIZE, height=IMAGE_SIZE)
        time.sleep(0.02)
        cmd.rotate('z', -90)
        return (png_path_x, png_path_y, png_path_z)

    def get_positions():
        model = cmd.get_model('all', 1)
        positions = np.array(model.get_coord_list())
        return tf.convert_to_tensor(positions, preferred_dtype=tf.float32)

    def get_atom_features(atom):
        return np.array([ord(atom.chain.lower())/122,
                         atom.get_mass(),
                         atom.formal_charge,
                         atom.partial_charge,
                         atom.vdw,
                         atom.b,
                         atom.get_free_valence(0),
                         sum([ord(i) for i in atom.resi])//len(atom.resi),
                         sum([ord(i) for i in atom.resn])//len(atom.resn),
                         sum([ord(i) for i in atom.symbol])//len(atom.symbol)])

    def undock(self):
        for chain in self.chains:
            selection_string = 'chain ' + chain
            translation_vector = [
                random.randrange(MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE),
                random.randrange(MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE),
                random.randrange(MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE)]
            cmd.translate(translation_vector, selection_string)
            cmd.rotate('x', random.randrange(-360, 360), selection_string)
            cmd.rotate('y', random.randrange(-360, 360), selection_string)
            cmd.rotate('z', random.randrange(-360, 360), selection_string)

    def unfold(self):
        for chain in self.chains:
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
        except Exception:
            print('failed to set dihedral at ', name, index)

    def load(self):
        cmd.delete('all')
        if screenshot:
            episode_path = os.path.join(run_path, str(episode))
            episode_images_path = os.path.join(episode_path, 'images')
            episode_stacks_path = os.path.join(episode_path, 'stacks')
            episode_pngs_path = os.path.join(episode_images_path, 'pngs')
            for p in [episode_images_path, episode_stacks_path, episode_pngs_path]:
                os.makedirs(p)
        # get pdb
        if episode < len(pedagogy) and RANDOM_PROTEINS is False:
            try:
                pdb_name = pedagogy[(episode + 1) * BIGGEST_FIRST_IF_NEG]
            except Exception as e:
                print('error loading protein', e)
                pdb_name = random.choice(pedagogy)
        else:
            pdb_name = random.choice(pedagogy)
        pdb_file_name = pdb_name + '.pdb'
        pdb_path = os.path.join(ROOT, 'pdbs', pdb_file_name)
        print('loading', pdb_path)
        if not os.path.exists(pdb_path):
            cmd.fetch(pdb_name, path='./pdbs', type='pdb')
        elif os.path.exists(pdb_path):
            cmd.load(pdb_path)
        # setup task
        cmd.remove('solvent')
        cmd.select(name='current', selection='all')
        initial_positions = get_positions()
        initial_distances = calculate_distances(initial_positions)
        zeroes = tf.zeros_like(initial_positions)
        initial = tf.concat([initial_positions, zeroes], 1)
        num_atoms = int(initial_positions.shape[0])
        num_atoms_squared = num_atoms ** 2
        chains = cmd.get_chains('current')
        model = cmd.get_model('current', 1)
        features = np.array([get_atom_features(atom) for atom in model.atom])
        features = tf.convert_to_tensor(features, preferred_dtype=tf.float32)
        mass = np.array([atom.get_mass() for atom in model.atom])
        mass = tf.convert_to_tensor(mass, preferred_dtype=tf.float32)
        masses = tf.stack([mass, mass, mass], 1)
        undock(chains)
        unfold(chains)
        positions = get_positions()
        velocities = random_normal(positions.shape, 0, NOISE, 'float')
        current = tf.concat([positions, velocities], 1)
        print('')
        print('loaded', pdb_name, 'with', num_atoms, 'atoms')
        print('')
        return observation

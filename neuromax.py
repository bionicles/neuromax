# neuromax.py - why?: 1 simple file with functions over classes
from tensorflow.keras.layers import Input, Dense, Add, Concatenate, Activation
from tensorflow.keras.backend import random_normal
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
import tensorflow as tf
from pymol import cmd  # 2.1.1 conda
from PIL import Image, ImageDraw
import numpy as np
import imageio
import random
import shutil
import time
import csv
import os

# globals
velocities, masses, chains, model = [], [], [], []
episode_images_path, episode_stacks_path = '', ''
episode_path, run_path = '', ''
episode, step, num_atoms = 0, 0, 0
ROOT = os.path.abspath('.')
TIME = str(time.time())
atom_index = 0
pdb_name = ''
# task
IMAGE_SIZE = 256
POSITION_VELOCITY_LOSS_WEIGHT, SHAPE_LOSS_WEIGHT = 1, 1
MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE = 4, 16
MIN_STEPS_IN_UNDOCK, MAX_STEPS_IN_UNDOCK = 0, 5
MIN_STEPS_IN_UNFOLD, MAX_STEPS_IN_UNFOLD = 0, 1
SCREENSHOT_EVERY = 100
WARMUP = 1000
NOISE = 0.002
BUFFER = 42
# model
N_BLOCKS = 6
# training
STOP_LOSS_MULTIPLIER = 1.618
NUM_EPISODES = 1000
NUM_STEPS = 100


def make_block(units, features, MaybeNoiseOrOutput):
    block_output = Concatenate(2)([features, MaybeNoiseOrOutput])
    block_output = Activation('tanh')(block_output)
    block_output = Dense(units, 'tanh')(block_output)
    block_output = Dense(MaybeNoiseOrOutput.shape[-1], 'tanh')(block_output)
    block_output = Add()([block_output, MaybeNoiseOrOutput])
    return block_output


def make_resnet(name, in1, in2):
    features = Input((None, in1))
    noise = Input((None, in2))
    units = 1 + (in1 + in2) ** 2
    output = make_block(units, features, noise)
    for i in range(1, N_BLOCKS):
        output = make_block(units, features, output)
    resnet = Model([features, noise], output)
    try:
        plot_model(resnet, name + '.png', show_shapes=True)
    except Exception:
        print('Failed to plot the model architecture')  # windows issue
    return resnet


def load_pedagogy():
    global pedagogy
    with open(os.path.join('.', 'pedagogy.csv')) as csvfile:
        reader = csv.reader(csvfile)
        results = []
        for row in reader:
            for item in row:
                item = item.strip()
                results.append(item)
        pedagogy = results


def prepare_pymol():
    cmd.remove('solvent')
    color_chainbow()
    cmd.set('depth_cue', 0)
    cmd.show('spheres')
    cmd.zoom('all', buffer=BUFFER, state=-1)
    cmd.center('all')
    cmd.bg_color('black')


def pick_colors(number_of_colors):
    colors = ['red', 'orange', 'yellow', 'green', 'forest', 'blue',
              'marine', 'magenta', 'deeppurple', 'cyan', 'brown', 'silver']
    return random.sample(colors, number_of_colors)


def color_chainbow():
    for i in range(len(chains)):
        cmd.color(pick_colors(1)[0], 'chain ' + chains[i])


def make_gif():
    gif_name = '{}-{}-{}.gif'.format(episode, pdb_name, TIME)
    gif_path = os.path.join(episode_path, gif_name)
    print("episode path", episode_path)
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
                     atom.q,
                     atom.b,
                     atom.get_free_valence(0),
                     sum([ord(i) for i in atom.resi])//len(atom.resi),
                     sum([ord(i) for i in atom.resn])//len(atom.resn),
                     sum([ord(i) for i in atom.symbol])//len(atom.symbol)])


def undock(chains):
    global step
    steps_in_undock = random.randint(MIN_STEPS_IN_UNDOCK, MAX_STEPS_IN_UNDOCK)
    print('undocking', pdb_name, steps_in_undock, 'times')
    step_vector_array, sum_vector = [], {}
    for undock_step in range(steps_in_undock):
        current_step_vectors = []
        for chain in chains:  # x, y, z, rx, ry, rz
            vector = np.array([
                random.randrange(MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE),
                random.randrange(MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE),
                random.randrange(MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE),
                random.randrange(-360, 360),
                random.randrange(-360, 360),
                random.randrange(-360, 360)])
            # save the sum (final destination)
            if chain not in sum_vector.keys():
                sum_vector[chain] = vector
            else:
                sum_vector[chain] = sum_vector[chain] + vector
            current_step_vectors.append(vector)
        step_vector_array.append(current_step_vectors)
    # move to the final position
    for chain in chains:
        if chain in sum_vector.keys():
            selection_string = 'chain ' + chain
            vector = sum_vector[chain]
            final_translation_vector = list(vector[:3])
            cmd.translate(final_translation_vector, selection_string)
            cmd.rotate('x', vector[3], selection_string)
            cmd.rotate('y', vector[4], selection_string)
            cmd.rotate('z', vector[5], selection_string)
    # set the zoom on the final destination
    prepare_pymol()
    # prepare to animate undock by moving back to original position
    for chain in chains:
        if chain in sum_vector.keys():
            selection_string = 'chain ' + chain
            vector = sum_vector[chain] * -1  # move back
            inverse_translation_vector = list(vector[:3])
            cmd.translate(inverse_translation_vector, selection_string)
            cmd.rotate('x', vector[3], selection_string)
            cmd.rotate('y', vector[4], selection_string)
            cmd.rotate('z', vector[5], selection_string)
    # move through the list of steps, move the chains, take the screenshots
    if screenshot:
        make_image()
    step += 1
    for step_vector in step_vector_array:
        for k in range(len(step_vector)):
            chain_vector = step_vector[k]
            chain = chains[k]
            current_vector = list(chain_vector[:3])
            cmd.translate(current_vector, 'current and chain ' + chain)
            cmd.rotate('x', chain_vector[3], 'current and chain ' + chain)
            cmd.rotate('y', chain_vector[4], 'current and chain ' + chain)
            cmd.rotate('z', chain_vector[5], 'current and chain ' + chain)
        if screenshot:
            make_image()
        step += 1


def unfold(chains):
    global step
    num_steps = random.randint(MIN_STEPS_IN_UNFOLD, MAX_STEPS_IN_UNFOLD)
    print('unfolding', pdb_name, num_steps, 'times')
    for fold_step in range(num_steps):
        for chain in chains:
            np.array([unfold_index(name, index) for name, index in
                      cmd.index('byca (chain {})'.format(chain))])
        if screenshot:
            cmd.unpick()
            make_image()
        step += 1


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


def reset():
    cmd.delete('all')
    global positions, initial_distances, velocities, chains, model, pdb_name
    global masses, num_atoms
    if screenshot:
        global episode_path, episode_images_path
        global episode_stacks_path, episode_pngs_path
        episode_path = os.path.join(run_path, str(episode))
        episode_images_path = os.path.join(episode_path, 'images')
        episode_stacks_path = os.path.join(episode_path, 'stacks')
        episode_pngs_path = os.path.join(episode_images_path, 'pngs')
        for p in [episode_images_path, episode_stacks_path, episode_pngs_path]:
            os.makedirs(p)
    # get pdb
    if episode == 0:
        pdb_name = pedagogy[-1]
    elif episode < len(pedagogy):
        pdb_name = pedagogy[episode - 1]
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
    num_atoms = initial_positions.shape[0]
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
    print("")
    print("loaded", pdb_name, "with", num_atoms, "atoms")
    print("")
    return initial, current, features


def move_atom(xyz):
    global atom_index
    atom_selection_string = 'id ' + str(atom_index)
    xyz = xyz.numpy().tolist()
    cmd.translate(xyz, atom_selection_string)
    atom_index += 1


def move_atoms(force_field):
    global positions, velocities, atom_index
    acceleration = force_field / masses
    noise = random_normal((num_atoms, 3), 0, NOISE, dtype="float32")
    velocities += acceleration + noise
    iter_velocity = tf.squeeze(velocities)
    atom_index = 0
    np.array([move_atom(xyz) for xyz in iter_velocity])
    positions = tf.math.add(positions, velocities)
    return positions, velocities


def calculate_distances(position_tensor):
    distances = tf.reduce_sum(position_tensor * position_tensor, 1)
    distances = tf.reshape(distances, [-1, 1])
    distances = distances - 2 * tf.matmul(
        position_tensor,
        tf.transpose(position_tensor)) + tf.transpose(distances)
    return distances


def loss(action, initial):
    global current
    position, velocities = move_atoms(action)
    # meta distance (shape loss)
    current_distances = calculate_distances(position)
    shape_loss = tf.losses.mean_squared_error(current_distances, initial_distances)
    # normal distance (position + velocity loss)
    current = tf.concat([positions, velocities], 1)
    position_velocity_loss = tf.losses.mean_squared_error(current, initial)
    # loss value (sum)
    loss_value = shape_loss * SHAPE_LOSS_WEIGHT
    loss_value += position_velocity_loss * POSITION_VELOCITY_LOSS_WEIGHT
    print("")
    print('model', TIME, 'episode', episode, 'step', step)
    print('shape loss', shape_loss.numpy().tolist())
    print('loss', position_velocity_loss.numpy().tolist())
    print('total loss', loss_value.numpy().tolist())
    return loss_value


def train():
    global run_path, screenshot, positions, velocities, features, episode, step
    run_path = os.path.join(ROOT, 'runs', TIME)
    os.makedirs(run_path)
    save_path = os.path.join(run_path, 'model.h5')
    actor = make_resnet('actor', 17, 3)
    adam = tf.train.AdamOptimizer()
    episode = 0
    load_pedagogy()
    for i in range(NUM_EPISODES):
        print("")
        print("BEGIN EPISODE", episode)
        print("")
        screenshot = episode > WARMUP and episode % SCREENSHOT_EVERY == 0
        done, step = False, 0
        initial, current, features = reset()
        initial_loss = loss(tf.zeros_like(positions), initial)
        stop_loss = initial_loss * STOP_LOSS_MULTIPLIER
        while not done:
            with tf.GradientTape() as tape:
                atoms = tf.expand_dims(tf.concat([current, features], 1), 0)
                force_field = tf.expand_dims(random_normal((num_atoms, 3)), 0)
                force_field = tf.squeeze(actor([atoms, force_field]) * -1, 0)
                loss_value = loss(force_field, initial)
            gradients = tape.gradient(loss_value, actor.trainable_weights)
            adam.apply_gradients(zip(gradients, actor.trainable_weights))
            if screenshot:
                make_image()
            step += 1
            done_because_step = step > NUM_STEPS
            done_because_loss = loss_value > stop_loss
            done = done_because_step or done_because_loss
        reason = 'STEP' if done_because_step else 'STOP LOSS'
        print('done because of', reason)
        if screenshot:
            make_gif()
        episode += 1
        actor.save(save_path)


if __name__ == '__main__':
    tf.enable_eager_execution()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    train()

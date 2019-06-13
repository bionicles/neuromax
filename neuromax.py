# neuromax.py - why?: 1 simple file with functions over classes
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
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
import time
import csv
import os

tf.enable_eager_execution()
# globals
initial, current, positions, velocities, features = [], [], [], [], []
masses, chains, model = [], [], []
episode_stacks_path, episode_pngs_path = '', ''
episode, step, num_atoms = 0, 0, 0
ROOT = os.path.abspath('.')
TIME = str(time.time())
pdb_name = ''
# task
MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE = 4, 8
MIN_STEPS_IN_UNDOCK, MAX_STEPS_IN_UNDOCK = 3, 4
MIN_STEPS_IN_UNFOLD, MAX_STEPS_IN_UNFOLD = 0, 1
STOP_LOSS_MULTIPLE = 10
SCREENSHOT_EVERY = 2
IMAGE_SIZE = 256
NUM_EPISODES = 1
NUM_STEPS = 10
BUFFER = 64
# model
N_BLOCKS = 0
# training
PATIENCE = 9
EPOCHS = 10

def make_block(block_inputs, MaybeNoiseOrOutput):
    block_output = Concatenate(2)([block_inputs, MaybeNoiseOrOutput])
    block_output = Activation("tanh")(block_output)
    block_output = Dense(MaybeNoiseOrOutput.shape[-1], 'tanh')(block_output)
    block_output = Dense(MaybeNoiseOrOutput.shape[-1], 'tanh')(block_output)
    block_output = Add()([block_output, MaybeNoiseOrOutput])
    return block_output


def make_resnet(name, in1, in2):
    print("MAKE RESNET", name)
    features = Input((None, in1))
    noise = Input((None, in2))
    output = make_block(features, noise)
    for i in range(1, N_BLOCKS):
        output = make_block(features, output)
    resnet = Model([features, noise], output)
    try:
        plot_model(resnet, name + '.png', show_shapes=True)
    except Exception:
        print("Failed to plot the model architecture")  # windows issue
    return resnet


def load_pedagogy():
    global pedagogy
    with open('./csvs/pedagogy.csv') as csvfile:
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
    cmd.show('surface')
    cmd.zoom('all', buffer=BUFFER, state=-1)
    cmd.center('all')
    cmd.bg_color('black')


def pick_colors(number_of_colors):
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'marine', 'violet']
    return random.sample(colors, number_of_colors)


def color_chainbow():
    colors = pick_colors(len(chains))
    for i in range(len(chains)):
        cmd.color(colors[i], 'chain ' + chains[i])


def make_gif():
    gif_name = 'r-{}_e-{}_p-{}.gif'.format(TIME, episode, pdb_name)
    gif_path = os.path.join(ROOT, 'gifs', gif_name)
    imagepaths, images = [], []
    for stackname in os.listdir(episode_stacks_path):
        print("processing", stackname)
        filepath = os.path.join(episode_stacks_path, stackname)
        imagepaths.append(filepath)
    imagepaths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for imagepath in imagepaths:
        image = imageio.imread(imagepath)
        images.append(image)
    print('saving gif to', gif_path)
    imageio.mimsave(gif_path, images)


def make_image():
    step_string = str(step)
    png_paths = take_pictures(step_string)
    images = [np.asarray(Image.open(png)) for png in png_paths]
    vstack = np.vstack(images)
    image_path = os.path.join(episode_stacks_path, step_string + '.png')
    vstack_img = Image.fromarray(vstack)
    x,y = vstack_img.size
    draw = ImageDraw.Draw(vstack_img)
    model_id_string = "Bit Pharma Neuromax: " + TIME
    x3, y3 = draw.textsize(model_id_string)
    draw.text(((x-x3)/2, y-42),model_id_string,(255,255,255))
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
    return positions


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


def undock():
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


def unfold():
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
    global chains, model, pdb_name, masses, num_atoms, initial
    global initial, current, positions, velocities, features
    if screenshot:
        global episode_stacks_path, episode_pngs_path
        episode_images_path = os.path.join(ROOT, 'images', TIME, str(episode))
        os.makedirs(episode_images_path)
        episode_stacks_path = os.path.join(episode_images_path, 'stacks')
        os.makedirs(episode_stacks_path)
        episode_pngs_path = os.path.join(episode_images_path, 'pngs')
        os.makedirs(episode_pngs_path)
    # get pdb
    pdb_name = pedagogy[episode] if episode < len(pedagogy) else random.choice(pedagogy)
    pdb_file_name = pdb_name + '.pdb'
    pdb_path = os.path.join(ROOT, 'pdbs', pdb_file_name)
    print('loading', pdb_path)
    if not os.path.exists(pdb_path):
        cmd.fetch(pdb_name, path="./pdbs", type='pdb')
    elif os.path.exists(pdb_path):
        cmd.load(pdb_path)
    # setup task
    cmd.remove('solvent')
    cmd.select(name='current', selection='all')
    initial_positions = get_positions()
    velocities = np.random.normal(size=initial_positions.shape)
    initial = np.stack([initial_positions, np.zeros_like(velocities)], 1)
    print("INITIAL SHAPE", initial.size)
    num_atoms = initial_positions.shape[0]
    chains = cmd.get_chains('current')
    model = cmd.get_model('current', 1)
    features = np.array([get_atom_features(atom) for atom in model.atom])
    mass = np.array([atom.get_mass() for atom in model.atom])
    masses = np.float32(np.stack([mass, mass, mass], 1))
    undock()
    unfold()
    positions = get_positions()
    initial = np.stack([positions, velocities], 1)
    initial = np.expand_dims(np.float_32(initial), 0)
    current = np.expand_dims(np.float_32(current), 0)
    features = np.expand_dims(np.float32(features), 0)


def move_atom(xyz, atom_index):
    atom_selection_string = 'id ' + str(atom_index)
    xyz = list(xyz)
    cmd.translate(xyz, atom_selection_string)
    atom_index += 1


def move_atoms(potentials):
    global positions, velocities
    force = -1 * potentials
    acceleration = force / masses
    noise = random_normal(potentials.shape)
    new_velocities = tf.add_n([velocities, acceleration, noise])
    new_velocities = tf.squeeze(new_velocities)
    atom_index = 0
    np.array([move_atom(xyz, atom_index) for xyz in new_velocities])
    positions = positions + new_velocities
    velocities = tf.expand_dims(new_velocities, 0)
    return positions, velocities


def loss(action, perfection):
    global current
    positions, velocities = move_atoms(action)
    current = tf.concat(2, [positions, velocities])
    return tf.losses.mean_squared_error(current, initial)


def train():
    global screenshot, positions, velocities, features, episode, step, work
    callbacks = [TensorBoard(), ReduceLROnPlateau(monitor='loss'),
                 EarlyStopping(monitor="loss", patience=PATIENCE)]
    actor = make_resnet('actor', 17, 3)
    adam = tf.train.AdamOptimizer()
    episode = 0
    load_pedagogy()
    actor.compile(loss=loss, optimizer=adam)
    for i in range(NUM_EPISODES):
        screenshot = episode % SCREENSHOT_EVERY == 0
        done, step = False, 0
        reset()
        while not done:
            atoms = tf.concat([positions, velocities, features], 2)
            actor.fit(x=atoms, y=initial, epochs=EPOCHS, batch_size=num_atoms,
                      callbacks=callbacks, verbose=1)
            if screenshot:
                make_image()
            step += 1
        if screenshot:
            make_gif()
        actor.save_model()
        episode += 1


if __name__ == '__main__':
    train()

# neuromax.py - why?: 1 simple file with functions over classes
from tf.keras.callbacks import TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from scipy.spatial.distance import cdist
from tf.keras.backend import stack
from pymol import cmd
from PIL import Image
import numpy as np
import imageio
import random
import time
import csv
import os


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


# globals
TIMESTAMP = str(time.time())
ROOT = os.path.abspath('.')
episode_stacks_path = ''
episode_pngs_path = ''
current_positions = []
initial_positions = []
transpose_masses = []
velocity_zero = []
max_work = np.inf
pdb_name = ''
masses = []
chains = []
episode = 0
step = 0


# model
BLOCKS_PER_RESNET = 2
# predictor (N, 17) -> (N, 6) x y z vx vy vz
PREDICT_PLAN = [
    AttrDict({
        'type': Dense,
        'units': 17,
        'fn': 'tanh'}),
    AttrDict({
        'type': Dense,
        'units': 6,
        'fn': 'tanh'})]
# actor (N, 17 + 6 = 23) -> (N, 3) x y z
ACTOR_PLAN = [
    AttrDict({
        'type': Dense,
        'units': 23,
        'fn': 'selu'}),
    AttrDict({
        'type': Dense,
        'units': 3,
        'fn': 'tanh'})]
# critic (N, 17 + 6 + 3 = 26) -> (1) reward
CRITIC_PLAN = [
    AttrDict({
        'type': Dense,
        'units': 26,
        'fn': 'tanh'}),
    AttrDict({
        'type': Dense,
        'units': 1,
        'fn': 'tanh'})]


def make_layer(recipe, tensor):
    layer = recipe.type(units=recipe.units, activation=recipe.activation)
    return layer(tensor)


def make_block(array, prior):
    output = make_layer(array[0], prior)
    for i in range(1, len(array)):
        output = make_layer(array[i], output)
    return Add([output, prior])


def make_resnet(input_shape, recipe, name):
    input = Input(input_shape)
    output = make_block(recipe, input)
    for block in range(1, BLOCKS_PER_RESNET):
        output = make_block(recipe, output)
    resnet = Model(input, output)
    plot_model(resnet, name, show_shapes=True)
    return resnet, output


def make_agent():
    predictor, prediction = make_resnet((None, 17), PREDICT_PLAN, 'predictor')
    actor_input = stack([predictor.input, prediction])
    actor, action = make_resnet(actor_input, ACTOR_PLAN, 'actor')
    critic_input = stack([actor.input, prediction])
    critic, criticism = make_resnet(critic_input, CRITIC_PLAN, 'critic')
    agent = Model(predictor.input, [prediction, action, criticism])
    plot_model(agent, 'agent.png', show_shapes=True)
    return predictor, actor, critic, agent


# task
MAX_UNDOCK_DISTANCE = 100
MIN_UNDOCK_DISTANCE = 10
MAX_STEPS_IN_UNDOCK = 3
MIN_STEPS_IN_UNDOCK = 2
STOP_LOSS_MULTIPLE = 10
SCREENSHOT_EVERY = 20
NUM_EPISODES = 10000
NUM_FEATURES = 17
IMAGE_SIZE = 256
ATOM_JIGGLE = 1


def load_pedagogy():
    with open('./csvs/pedagogy.csv') as csvfile:
        reader = csv.reader(csvfile)
        results = []
        for row in reader:
            for item in row:
                item = item.strip()
                results.append(item)
        return results


def prepare_pymol():
    cmd.remove('solvent')
    color_chainbow()
    cmd.set('depth_cue', 0)
    cmd.unpick()
    cmd.show('surface')
    cmd.zoom('all', buffer=42, state=-1)
    cmd.center('all')
    cmd.bg_color('black')
    cmd.deselect()


def color_chainbow(chains):
    colors = pick_colors(len(chains))
    for i in range(len(chains)):
        cmd.color(colors[i], 'chain ' + chains[i])


def pick_colors(number_of_colors):
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'marine', 'violet']
    return random.sample(colors, number_of_colors)


def make_gif():
    gif_name = 'r-{}_e-{}_p-{}.gif'.format(TIMESTAMP, episode, pdb_name)
    gif_path = os.path.join(ROOT, 'gifs', gif_name)
    imagepaths = []
    images = []
    for stackname in os.listdir(episode_stacks_path):
        filepath = os.path.join(episode_stacks_path, stackname)
        imagepaths.append(filepath)
    imagepaths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    del imagepaths[0]  # delete stacked images folder from the list
    for imagepath in imagepaths:
        image = imageio.imread(imagepath)
        images.append(image)
    print('saving gif to', gif_path)
    imageio.mimsave(gif_path, images)
    return gif_path, gif_name


def make_image():
    step_indicator = 'e-{}_p-{}_s-{}'.format(episode, pdb_name, step)
    png_paths = take_pictures(step_indicator)
    images = [np.asarray(Image.open(png)) for png in png_paths]
    vstack = np.vstack(images)
    image_path = os.path.join(episode_stacks_path, pdb_name + '.png')
    vstack_img = Image.fromarray(vstack)
    vstack_img.save(image_path)


def take_pictures(step_indicator):
    print('screenshot episode', episode, 'pdb', pdb_name, 'step', step)
    global episode_pngs_path
    cmd.deselect()
    png_path_x = os.path.join(episode_pngs_path, step_indicator + '-X.png')
    cmd.png(png_path_x, width=IMAGE_SIZE, height=IMAGE_SIZE)
    time.sleep(0.02)
    cmd.rotate('y', 90)
    png_path_y = os.path.join(episode_pngs_path, step_indicator + '-Y.png')
    cmd.png(png_path_y, width=IMAGE_SIZE, height=IMAGE_SIZE)
    time.sleep(0.02)
    cmd.rotate('y', -90)
    cmd.rotate('z', 90)
    png_path_z = os.path.join(episode_pngs_path, step_indicator + '-Z.png')
    cmd.png(png_path_z, width=IMAGE_SIZE, height=IMAGE_SIZE)
    time.sleep(0.02)
    cmd.rotate('z', -90)
    return (png_path_x, png_path_y, png_path_z)


def get_positions():
    model = cmd.get_model('current', 1)
    return np.array(model.get_coord_list())


def get_atom_features(atom):
    return np.array([
        ord(atom.chain.lower())/122,
        atom.get_mass(),
        atom.formal_charge,
        atom.partial_charge,
        atom.vdw,
        atom.q,
        atom.b,
        atom.get_free_valence(0),
        sum([ord(i) for i in atom.resi])//len(atom.resi),
        sum([ord(i) for i in atom.resn])//len(atom.resn),
        sum([ord(i) for i in atom.symbol])//len(atom.symbol)
    ])


def calculate_work():
    distance = cdist(current_positions, initial_positions)
    # work to move atoms back to initial positions
    work = distance * transpose_masses
    work = np.triu(work, 1)  # k=1 should zero main diagonal
    work = np.sum(work)
    # work to slow atoms to a stop
    work_to_stop = np.transpose(velocities) * transpose_masses
    work_to_stop = np.triu(work_to_stop, 1)
    work_to_stop = np.sum(work_to_stop)
    if work_to_stop > 0:
        work += work_to_stop
    return work


def undock(screenshotting):
    steps_in_undock = random.randint(MIN_STEPS_IN_UNDOCK, MAX_STEPS_IN_UNDOCK)
    print('undocking', pdb_name, steps_in_undock, 'times')
    step_vector_array = []
    sum_vector = {}
    # make a list of steps
    for step in range(steps_in_undock):
        current_step_vectors = []
        for chain in chains:
            # x, y, z, rx, ry, rz
            vector = np.array([
                random.randrange(MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE),
                random.randrange(MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE),
                random.randrange(MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE),
                random.randrange(MIN_UNDOCK_DISTANCE, MAX_UNDOCK_DISTANCE),
                random.randrange(-360, 360),
                random.randrange(-360, 360),
                random.randrange(-360, 360)
                ])
            # save the sum (final destination)
            if chain not in sum_vector.keys():
                sum_vector[chain] = vector
            else:
                sum_vector[chain] = sum_vector[chain] + vector
            # save the components also
            current_step_vectors.append(vector)
        step_vector_array.append(current_step_vectors)
    # move to the final position
    for chain in chains:
        if chain in sum_vector.keys():
            selection_string = 'current and chain ' + chain
            vector = sum_vector[chain]
            final_translation_vector = list(vector[:3])
            cmd.translate(final_translation_vector, selection_string)
            cmd.rotate('x', vector[3], selection_string)
            cmd.rotate('y', vector[4], selection_string)
            cmd.rotate('z', vector[5], selection_string)
    # set the zoom on the final destination
    if screenshotting:
        prepare_pymol()
    # prepare to animate undock by moving back to original position
    for chain in chains:
        if chain in sum_vector.keys():
            vector = sum_vector[chain] * -1  # move back
            inverse_translation_vector = list(vector[:3])
            cmd.translate(inverse_translation_vector, 'chain ' + chain)
            cmd.rotate('x', vector[3], 'current and chain ' + chain)
            cmd.rotate('y', vector[4], 'current and chain ' + chain)
            cmd.rotate('z', vector[5], 'current and chain ' + chain)
    # move through the list of steps, move the chains, take the screenshots
    if screenshotting:
        make_image()
    step = 1
    for step_vector in step_vector_array:
        for k in range(len(step_vector)):
            chain_vector = step_vector[k]
            chain = chains[k]
            current_vector = list(chain_vector[:3])
            cmd.translate(current_vector, 'current and chain ' + chain)
            cmd.rotate('x', chain_vector[3], 'current and chain ' + chain)
            cmd.rotate('y', chain_vector[4], 'current and chain ' + chain)
            cmd.rotate('z', chain_vector[5], 'current and chain ' + chain)
        if screenshotting:
            make_image()
        step += 1


def unfold(screenshotting, chains):
    global step
    num_steps = random.randint(MIN_STEPS_IN_UNDOCK, MAX_STEPS_IN_UNDOCK)
    print('unfolding', pdb_name, num_steps, 'times')
    for fold_step in range(num_steps):
        for chain in chains:
            np.array([
                unfold_index(name, index) for name, index in
                cmd.index('byca (chain {})'.format(chain))])
        if screenshotting:
            make_image()
    step = step + num_steps


def unfold_index(name, index):
    selection_string_array = [
        'first (({}`{}) extend 2 and name C)'.format(name, index),  # prev C
        'first (({}`{}) extend 1 and name N)'.format(name, index),  # this N
        '({}`{})'.format(name, index),                              # this CA
        'last (({}`{}) extend 1 and name C)'.format(name, index),   # this C
        'last (({}`{}) extend 2 and name N)'.format(name, index),   # next N
    ]
    try:
        phi_random = random.randint(0, 360)
        cmd.set_dihedral(
            selection_string_array[0],
            selection_string_array[1],
            selection_string_array[2],
            selection_string_array[3],
            phi_random)
        psi_random = random.randint(0, 360)
        cmd.set_dihedral(
            selection_string_array[1],
            selection_string_array[2],
            selection_string_array[3],
            selection_string_array[4],
            psi_random)
    except:
        print('failed to set dihedral at ', name, index)


def reset():
    cmd.delete('all')
    # make paths
    global current_velocities
    global current_positions
    global initial_positions
    global screenshotting
    global pdb_name
    global max_work
    global episode
    if screenshotting:
        global episode_stacks_path
        global episode_pngs_path
        episode_images_path = os.path.join(
            ROOT, 'images', TIMESTAMP, str(episode))
        os.makedirs(episode_images_path)
        episode_stacks_path = os.path.join(episode_images_path, 'stacks')
        os.makedirs(episode_stacks_path)
        episode_pngs_path = os.path.join(episode_images_path, 'pngs')
        os.makedirs(episode_pngs_path)
    # get pdb
    pdb_name = random.choice(pedagogy)
    pdb_file_name = pdb_name + '.pdb'
    pdb_path = os.path.join(ROOT, 'pdbs', pdb_file_name)
    print('loading', pdb_path)
    if not os.path.exists(pdb_path):
        cmd.fetch(pdb_name, path=pdb_path, type='pdb')
    elif os.path.exists(pdb_path):
        cmd.load(pdb_path)
    # setup task
    cmd.remove('solvent')
    cmd.select(name='current', selection='all')
    initial_positions = get_positions()
    current_velocities = np.random.normal(size=initial_positions.shape)
    chains = cmd.get_chains('current')
    model = cmd.get_model('current', 1)
    features = np.array([get_atom_features(atom) for atom in model.atom])
    masses = np.array([atom.get_mass() for atom in model.atom])
    masses = np.array([masses, masses, masses])
    transpose_masses = np.transpose(masses)
    undock()
    unfold()
    current_positions = get_positions()
    initial_work = calculate_work()
    max_work = initial_work * STOP_LOSS_MULTIPLE
    return (
        initial_positions,
        current_positions,
        current_velocities,
        transpose_masses,
        features,
        masses,
        chains,
        max_work)


def move_atom(xyz, atom_index):
    xyz = list(xyz)
    atom_selection_string = 'id ' + str(atom_index)
    cmd.translate(xyz, atom_selection_string)
    atom_index += 1


def step(potentials):
    force = -1 * potentials
    acceleration = force / transpose_masses
    noise = np.random.normal(loc=0.0, scale=ATOM_JIGGLE, size=velocities.shape)
    velocities += acceleration + noise
    atom_index = 0
    np.array([move_atom(xyz, atom_index) for xyz in velocities])
    new_positions = current_positions + velocities
    work = calculate_work()
    done = work > max_work
    state = stack(new_positions, velocities, features)
    return state, work, done


def train():
    global initial_positions
    global current_positions
    global screenshotting
    global current_velocities
    global velocities
    global features
    global pedagogy
    global episode
    global step
    callbacks = [TensorBoard(), ReduceLROnPlateau(monitor='loss')]
    predictor, actor, critic, agent = make_agent()
    predictor.compile(loss='mse', optimizer='nadam')
    critic.compile(loss='mse', optimizer='nadam')
    actor.compile(loss='mse', optimizer='nadam')
    pedagogy = load_pedagogy()
    episode = 0
    memory = []
    for i in range(NUM_EPISODES):
        screenshotting = episode % SCREENSHOT_EVERY == 0
        initial_positions, current_positions, features = reset()
        zero_velocities = np.zeros_like(initial_positions)
        done = False
        step = 0
        while not done:
            state = stack(current_positions, current_velocities, features)
            prediction, action, criticism = agent(state)
            new_positions, new_velocities, reward, done = step(action)
            memory.append(AttrDict({
                'current_positions': current_positions,
                'new_positions': new_positions,
                'current_velocities': current_velocities,
                'new_velocities': new_velocities,
                'prediction': prediction,
                'criticism': criticism,
                'action': action,
                'reward': reward
            }))
            current_velocities = new_velocities
            current_positions = new_positions
            step += 1
        make_gif()
        # optimize when done
        for event in memory:
            action_result = stack(event.new_positions, event.new_velocities)
            action_target = stack(initial_positions, zero_velocities)
            actor.fit(action_result, action_target, callbacks=callbacks)
            future = stack(event.new_positions, event.new_velocities)
            predictor.fit(event.prediction, future, callbacks=callbacks)
            critic.fit(event.criticism, event.reward, callback=callbacks)
        agent.save_model()
        episode += 1


if __name__ == '__main__':
    train()

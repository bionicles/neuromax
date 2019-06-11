# neuromax.py - why?: 1 simple file with functions over classes
from tensorflow.keras.layers import Input, Dense, add, concatenate
from scipy.spatial.distance import cdist
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
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

TIMESTAMP = str(time.time())
ROOT = os.path.abspath(".")
# model
BLOCKS_PER_RESNET = 4
LAYERS_PER_BLOCK = 2
# predictor :: input => position+velocity prediction shape(num_atoms, 6)
PREDICTOR_INPUT_SHAPE = 17
UNITS_PREDICTOR = 500
PREDICTOR_RECIPE = [(UNITS_PREDICTOR, "tanh"), (17*2, "relu")]
# actor     :: input, prediction => potentials shape(None, 3)
ACTOR_INPUT_SHAPE = 17*2
UNITS_ACTOR = 500
ACTOR_RECIPE = [(UNITS_ACTOR, "selu"), (3, "tanh")]
# critic    :: state, prediction, action => reward prediction shape(1)
CRITIC_INPUT_SHAPE = 17 + 6 + 3
UNITS_CRITIC = 500
CRITIC_RECIPE = [(UNITS_CRITIC, "tanh"), (1, "tanh")]
# task
MAX_UNDOCK_DISTANCE = 100
MIN_UNDOCK_DISTANCE = 10
MAX_STEPS_IN_UNDOCK = 3
MIN_STEPS_IN_UNDOCK = 2
STOP_LOSS_MULTIPLE = 10
SCREENSHOT_EVERY = 20
IMAGE_SIZE = 256
ATOM_JIGGLE = 1

def load_pedagogy():
    csvpath = os.path.join(ROOT, "csvs/pedagogy.csv")
    results = []
    with open(csvpath) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for item in row:
                item = item.strip()
                results.append(item)
    return results

pedagogy = load_pedagogy()

def prepare_pymol():
    cmd.remove("solvent")
    color_chainbow()
    cmd.set("depth_cue", 0)
    cmd.unpick()
    cmd.show('surface')
    cmd.zoom("all", buffer=42, state=-1)
    cmd.center("all")
    cmd.bg_color("black")
    cmd.deselect()

def color_chainbow(chains):
    colors = pick_colors(len(chains))
    for i in range(len(chains)):
        cmd.color(colors[i], "chain " + chains[i])

def pick_colors(number_of_colors):
    colors = ["red", "orange", "yellow", "green", "blue", "marine", "violet"]
    return random.sample(colors, number_of_colors)

def make_gif():
    gif_name = "r-{}_e-{}_p-{}.gif".format(TIMESTAMP, episode, pdb_name)
    gif_path = os.path.join(ROOT, "gifs", gif_name)
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
    print("saving gif to", gif_path)
    imageio.mimsave(gif_path, images)
    return gif_path, gif_name

def make_image():
    step_indicator = "e-{}_p-{}_s-{}".format(episode, pdb_name, step_number)
    png_paths = take_pictures(step_indicator)
    images = [np.asarray(Image.open(png)) for png in png_paths]
    vstack = np.vstack(images)
    image_path = os.path.join(
        episode_images_path, "stacks", step_indicator + ".png")
    vstack_img = Image.fromarray(vstack)
    vstack_img.save(image_path)

def take_pictures(step_indicator):
    cmd.deselect()
    print("screenshot episode", episode, "pdb", pdb_name, "step", step_number)
    png_path_x = os.path.join(episode_images_path, step_indicator + "-X.png")
    cmd.png(png_path_x, width=IMAGE_SIZE, height=IMAGE_SIZE)
    time.sleep(0.02)
    cmd.rotate('y', 90)
    png_path_y = os.path.join(episode_images_path, step_indicator + "-Y.png")
    cmd.png(png_path_y, width=IMAGE_SIZE, height=IMAGE_SIZE)
    time.sleep(0.02)
    cmd.rotate('y', -90)
    cmd.rotate('z', 90)
    png_path_z = os.path.join(episode_images_path, step_indicator + "-Z.png")
    cmd.png(png_path_z, width=IMAGE_SIZE, height=IMAGE_SIZE)
    time.sleep(0.02)
    cmd.rotate('z', -90)
    return (png_path_x, png_path_y, png_path_z)

def get_positions():
    model = cmd.get_model("current", 1)
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
    print("undocking", pdb_name, steps_in_undock, "times")
    step_vector_array = []
    sum_vector = {}
    # make a list of steps
    for step in range(steps_in_undock):
        current_step_vectors = []
        for chain in chains:
            # x, y, z, rx, ry, rz
            vector = np.array([
                random.randrange(MIN_UNDOCK, MAX_UNDOCK),
                random.randrange(MIN_UNDOCK, MAX_UNDOCK),
                random.randrange(MIN_UNDOCK, MAX_UNDOCK),
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
            selection_string = "current and chain " + chain
            vector = sum_vector[chain]
            final_translation_vector = list(vector[:3])
            cmd.translate(final_translation_vector, selection_string)
            cmd.rotate("x", vector[3], selection_string)
            cmd.rotate("y", vector[4], selection_string)
            cmd.rotate("z", vector[5], selection_string)
    # set the zoom on the final destination
    if screenshotting:
        prepare_pymol()
    # prepare to animate undock by moving back to original position
    for chain in chains:
        if chain in sum_vector.keys():
            vector = sum_vector[chain] * -1  # move back
            inverse_translation_vector = list(vector[:3])
            cmd.translate(inverse_translation_vector, "chain " + chain)
            cmd.rotate("x", vector[3], "current and chain " + chain)
            cmd.rotate("y", vector[4], "current and chain " + chain)
            cmd.rotate("z", vector[5], "current and chain " + chain)
    # move through the list of steps, move the chains, take the screenshots
    if screenshotting:
        make_image()
    step_number = 1
    for step_vector in step_vector_array:
        for k in range(len(step_vector)):
            chain_vector = step_vector[k]
            chain = chains[k]
            current_vector = list(chain_vector[:3])
            cmd.translate(current_vector, "current and chain " + chain)
            cmd.rotate("x", chain_vector[3], "current and chain " + chain)
            cmd.rotate("y", chain_vector[4], "current and chain " + chain)
            cmd.rotate("z", chain_vector[5], "current and chain " + chain)
        if screenshotting:
            make_image()
        step_number += 1

def unfold(screenshotting, chains):
    original_num_steps = num_steps
    num_steps = random.randint(MIN_STEPS_IN_UNDOCK, MAX_STEPS_IN_UNDOCK)
    print("unfolding", pdb_name, num_steps, "times")
    for step in range(num_steps):
        for chain in chains:
            np_array = np.array([unfold_index(name, index) for name, index in cmd.index("byca (chain {})".format(chain))])
        step = step + 1
        if screenshotting:
            prepare_pymol() # set the zoom on the final destination
            take_screenshot(params, pdb_name, num_proteins_seen, step)
    return num_steps + original_num_steps

def unfold_index(name, index):
    selection_string_array = [
        'first (({}`{}) extend 2 and name C)'.format(name, index), # prev C
        'first (({}`{}) extend 1 and name N)'.format(name, index), # this N
        '({}`{})'.format(name, index),                             # this CA
        'last (({}`{}) extend 1 and name C)'.format(name, index),  # this C
        'last (({}`{}) extend 2 and name N)'.format(name, index),  # next N
    ]
    try:
        phi_random = random.randint(0, 360)
        cmd.set_dihedral(selection_string_array[0], selection_string_array[1], selection_string_array[2], selection_string_array[3], phi_random)
        psi_random = random.randint(0, 360)
        cmd.set_dihedral(selection_string_array[1], selection_string_array[2], selection_string_array[3], selection_string_array[4], psi_random)
    except:
        print("failed to set dihedral at ", name, index)

def reset(screenshotting):
    episode += 1
    cmd.delete("all")
    # make paths
    if screenshotting:
        episode_images_path = os.path.join(ROOT, "images", TIMESTAMP, str(episode))
        os.makedirs(episode_images_path)
        episode_stacks_path = os.path.join(episode_images_path, "stacks")
        os.makedirs(episode_stacks_path)
        episode_pngs_path = os.path.join(episode_images_path, "pngs")
        os.makedirs(episode_pngs_path)
    # get pdb
    pdb_name = random.choice(pedagogy) + ".pdb"
    pdb_path = os.path.join(pdbs_path, pdb_name)
    print("loading", pdb_path)
    if not os.path.exists(pdb_path):
        cmd.fetch(pdb_name, path=pdb_path, type="pdb")
    elif os.path.exists(pdb_path):
        cmd.load(pdb_path)
    # setup task
    cmd.remove("solvent")
    cmd.select(name="current", selection="all")
    initial_positions = get_positions()
    velocities = np.random.normal(size=initial_positions.shape)
    chains = cmd.get_chains("current")
    step_number = 0
    model = cmd.get_model("current", 1)
    features = np.array([get_atom_features(atom) for atom in model.atom])
    masses = np.array([atom.get_mass() for atom in model.atom])
    masses = np.array([masses, masses, masses])
    transpose_masses = np.transpose(masses)
    undock(screenshotting)
    unfold(screenshotting)
    current_positions = get_positions()
    initial_work = calculate_reward()
    max_work = initial_work * STOP_LOSS_MULTIPLE
    return initial_positions, features, masses, transpose_masses, current_positions, max_work

def move_atom(xyz, atom_index):
    xyz = list(xyz)
    atom_selection_string = "id " + str(atom_index)
    cmd.translate(xyz, atom_selection_string)
    atom_index += 1

def step(potentials):
    force = -1 * potentials
    acceleration = force / transpose_masses
    noise = np.random.normal(loc=0.0, scale=ATOM_JIGGLE, size=velocities.shape)
    velocities += acceleration + noise
    atom_index = 0
    np.array([move_atom(xyz, atom_index) for xyz in velocities])
    positions += velocities
    reward = calculate_reward()
    done = work > max_work
    positions = get_positions()
    step_number += 1
    if done:
        make_gif()
    return state, reward, done

# BEGIN MODEL:
def make_layer(tuple, tensor):
    layer = Dense(units=tuple[0], activation=tuple[1])
    return layer(tensor)

def make_block(array, prior):
    output = make_layer(array[0], prior)
    for i in range(1, LAYERS_PER_BLOCK):
        output = make_layer(array[0], output)
    return add([output, prior])

def make_resnet(recipe, input, name, initial_input):
    output = make_block(recipe, input)
    for block in range(0, BLOCKS_PER_RESNET):
        output = make_block(recipe, output)
    output_tuple = recipe[1]
    output = Dense(units = output_tuple[0], activation = output_tuple[1])(output)
    resnet = Model(initial_input, output)
    plot_model(resnet, name, show_shapes = True)
    return resnet, output

def make_agent(num_features):
    predictor_input = Input((PREDICTOR_INPUT_SHAPE, ))
    predictor_first_layer = Dense(units = UNITS_PREDICTOR, activation = 'tanh')(predictor_input)
    predictor, prediction = make_resnet(PREDICTOR_RECIPE, predictor_first_layer,
                                                        name = 'predictor.png',
                                                        initial_input = input)
    actor_input = Input((ACTOR_INPUT_SHAPE, ))
    actor_first_layer = Dense(units = UNITS_PREDICTOR, activation = 'tanh')(actor_input)
    actor, action = make_resnet(ACTOR_RECIPE, actor_first_layer,
                                            name = 'actor.png',
                                            initial_input = actor_input)
    critic_input = Input((CRITIC_INPUT_SHAPE, ))
    critic_first_layer = Dense(units = UNITS_PREDICTOR, activation = 'tanh')(critic_input)
    critic, criticism = make_resnet(CRITIC_RECIPE, critic_first_layer,
                                                name = 'critic.png',
                                                initial_input =critic_input)
    agent_input = predictor.input

    agent = Model([predictor_input, ], [prediction, action, criticism])
    plot_model(agent, "agent.png",show_shapes=True)
    return predictor, actor, critic, agent

def train():
    predictor, actor, critic, agent = make_agent(17)
    predictor.compile(optimizer="adam")
    critic.compile(optimizer="adam")
    actor.compile(optimizer="adam")
    converged = false
    memory = []
    while not converged:
        screenshotting = episode % SCREENSHOT_EVERY == 0
        initial_positions, features, current_positions = reset(screenshotting)
        done = False
        while not done:
            prediction, action, criticism = agent(current_positions, features)
            next_positions, next_velocities, reward, done = step(action, screenshotting)
            event = AttrDict({
                "current_positions": current_positions,
                "next_velocities": next_velocities,
                "next_positions": next_positions,
                "prediction": prediction,
                "criticism": criticism,
                "action": action,
                "reward": reward
            })
            memory.append(event)
            current_observation = next_observation
        # optimize when done
        for event in memory:
            actor.fit(event.current_positions, event.initial_positions)
            prediction_target = concat(next_positions, next_velocities)
            predictor.fit(event.prediction, prediction_target)
            critic.fit(event.criticism, event.reward)
        agent.save_model()

if __name__ == "__main__":
    train()

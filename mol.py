from slackclient import SlackClient
from gym import spaces
from PIL import Image
import numpy as np
import imageio
import random
import scipy
import time
import gym
import csv
import os

from src.tasks.spaces import Array
from pymol import cmd


class PyMolEnv(gym.Env):
    def __init__(self, nurture, task):
        # setup state
        self.run_time_stamp = str(time.time())
        self.root = os.path.dirname(os.path.abspath("."))
        print("pymol root", self.root)
        self.task_root = self.root + "/metamage/src/tasks/molecules"
        print("pymol task_root", self.task_root)
        self.run_path = os.path.join(self.task_root, "runs", self.run_time_stamp)
        print("pymol run_path", self.run_path)
        os.makedirs(self.run_path)
        self.pedagogy = self.load_pedagogy()
        self.colors_chosen = []
        self.nurture = nurture
        self.episode = 0
        self.task = task
        # define spaces
        self.observation_space = spaces.Dict({
            "chains": Array(shape=(None, 6)),
            "aminos": Array(shape=(None, 7)),
            "bonds": Array(shape=(None, 1)),
            "image": Array(shape=(3, 128, 128)),
            "atoms": Array(shape=(None, 17))
        })
        self.action_space = spaces.Dict({
            "potentials": Array(shape=(None, 3))
        })

    def reset(self):
        self.episode += 1
        # clean up!
        cmd.delete("all")
        self.image_path = os.path.join(self.run_path, str(self.episode))
        os.makedirs(self.image_path)
        os.makedirs(self.image_path + "/stacks")
        os.makedirs(self.image_path + "/pngs")
        os.makedirs(self.image_path + "/jpgs")
        self.pdb = random.choice(self.pedagogy)
        self.pdb_folder_path = os.path.join(self.task_root, "inputs/pdbs/")
        self.pdb_path = self.pdb_folder_path + self.pdb + ".pdb"
        print("loading", self.pdb_path)
        if not os.path.exists(self.pdb_path):
            cmd.fetch(self.pdb, path=self.pdb_folder_path, type="pdb")
        elif os.path.exists(self.pdb_path):
            cmd.load(self.pdb_path)
        cmd.remove("solvent")
        # 5. prepare for task setup
        cmd.select(name="current", selection="all")
        self.get_metadata(self.task)
        if self.task is "dock":
            self.undock()
        if self.task is "fold":
            self.unfold()
        self.action_space = spaces.Dict({
            "potentials": Array(
                shape=self.velocities.shape,
                high=4.2,
                low=-4.2
                )
        })
        self.current = self.get_observation()
        # we need to know the initial work to calculate stop loss
        self.initial_work = self.calculate_reward()
        return self.current

    def undock(self):
        print("mol undock chains", self.chains)
        steps_in_undock = random.randint(
            self.nurture.min_steps_in_undock,
            self.nurture.max_steps_in_undock
            )
        print("undocking", self.pdb, steps_in_undock, "times")
        sum_vector = {}
        step_vector_array = []
        min_undock = self.nurture.min_undock
        max_undock = self.nurture.max_undock
        # make a list of steps
        for step in range(steps_in_undock):
            current_step_vectors = []
            for chain in self.chains:
                # x, y, z, rx, ry, rz
                vector = np.array([
                    random.randrange(min_undock, max_undock),
                    random.randrange(min_undock, max_undock),
                    random.randrange(min_undock, max_undock),
                    random.randrange(-360, 360),
                    random.randrange(-360, 360),
                    random.randrange(-360, 360)
                    ])
                # save the sum (final destination of undock)
                if chain not in sum_vector.keys():
                    sum_vector[chain] = vector
                else:
                    sum_vector[chain] = sum_vector[chain] + vector
                # save the components also
                current_step_vectors.append(vector)
            step_vector_array.append(current_step_vectors)
        # move to the final position
        for chain in self.chains:
            if chain in sum_vector.keys():
                selection_string = "current and chain " + chain
                vector = sum_vector[chain]
                final_translation_vector = list(vector[:3])
                cmd.translate(final_translation_vector, selection_string)
                cmd.rotate("x", vector[3], selection_string)
                cmd.rotate("y", vector[4], selection_string)
                cmd.rotate("z", vector[5], selection_string)
        # set the zoom on the final destination
        self.prepare_pymol()
        # prepare to animate undock by moving back to original position
        for chain in self.chains:
            if chain in sum_vector.keys():
                vector = sum_vector[chain] * -1  # move back
                inverse_translation_vector = list(vector[:3])
                cmd.translate(inverse_translation_vector, "chain " + chain)
                cmd.rotate("x", vector[3], "current and chain " + chain)
                cmd.rotate("y", vector[4], "current and chain " + chain)
                cmd.rotate("z", vector[5], "current and chain " + chain)
        # move through the list of steps, move the chains, take the screenshots
        self.get_image()
        self.step_number = 1
        for step_vector in step_vector_array:
            for k in range(len(step_vector)):
                chain_vector = step_vector[k]
                chain = self.chains[k]
                current_vector = list(chain_vector[:3])
                cmd.translate(current_vector, "current and chain " + chain)
                cmd.rotate("x", chain_vector[3], "current and chain " + chain)
                cmd.rotate("y", chain_vector[4], "current and chain " + chain)
                cmd.rotate("z", chain_vector[5], "current and chain " + chain)
            self.get_image()
            self.step_number += 1

    def unfold(self):
        print("task", self.task, "self.chains", self.chains)
        chain = self.chains[0]
        print("unfolding chain", chain)
        index_cmd_string = "current and byca (chain {})".format(chain)
        self.get_image()
        self.step_number += 1
        for name, index in cmd.index(index_cmd_string):
            selection_string_array = [
                'current and first (({}`{}) extend 2 and name C)'.format(name, index), # prev C
                'current and first (({}`{}) extend 1 and name N)'.format(name, index), # this N
                'current and ({}`{})'.format(name, index),                             # this CA
                'current and last (({}`{}) extend 1 and name C)'.format(name, index),  # this C
                'current and last (({}`{}) extend 2 and name N)'.format(name, index),  # next N
            ]
            try:
                phi_random = random.randint(-360, 360)
                cmd.set_dihedral(
                    selection_string_array[0],
                    selection_string_array[1],
                    selection_string_array[2],
                    selection_string_array[3],
                    phi_random
                )
            except:
                print("failed to set phi at: ", name, index)
            try:
                psi_random = random.randint(-360, 360)
                cmd.set_dihedral(
                    selection_string_array[1],
                    selection_string_array[2],
                    selection_string_array[3],
                    selection_string_array[4],
                    psi_random
                )
            except:
                print("failed to set psi at: ", name, index)
        self.get_image()

    def step(self, action):
        force = -1 * action["potentials"]
        acceleration = force / np.transpose(self.masses)
        noise = np.random.normal(
            loc=0.0,
            scale=self.nurture.atomic_noise,
            size=self.velocities.shape)
        print("acceleration dtype", acceleration.dtype, "noise dtype", noise.dtype)
        self.velocities += acceleration + noise
        self.atom_index = 0
        np.array([self.move_atom(xyz) for xyz in self.velocities])
        self.positions += self.velocities
        reward = self.calculate_reward()
        stop_loss = self.work > self.initial_work * self.nurture.stop_loss_multiple
        beyond_max_steps = self.step_number > self.nurture.max_steps
        done = stop_loss or beyond_max_steps
        observation = self.get_observation()
        info = None
        self.step_number += 1
        if done:
            print("done because beyond_max_steps", beyond_max_steps)
            print("done because stop loss", stop_loss)
            gif_path, gif_name = self.make_gif()
            self.slack_file(gif_path, gif_name)
        return observation, reward, done, info

    def move_atom(self, vector):
        movement_vector = list(vector)
        atom_selection_string = "id " + str(self.atom_index)
        cmd.translate(movement_vector, atom_selection_string)
        self.atom_index += 1

    def calculate_reward(self):
        distance = scipy.spatial.distance.cdist(
            self.positions,
            self.original_positions
            )
        # work to move the atoms to the proper spot
        mass = np.transpose(self.masses[0,:])
        self.work = distance * mass
        self.work = np.triu(self.work, 1)  # k=1 should zero main diagonal
        self.work = np.sum(self.work)
        # work to slow the atoms to a stop
        work_to_stop = np.transpose(self.velocities) * mass
        work_to_stop = np.triu(work_to_stop, 1)
        work_to_stop = np.sum(self.work)
        if work_to_stop > 0:
            self.work += work_to_stop
        # reward is the opposite of work
        return -1 * self.work

    def get_metadata(self, task):
        self.chains = cmd.get_chains("current")
        # we only fold 1 chain at a time for now:
        if self.task is "fold":
            chain_to_fold = random.choice(self.chains)
            for chain in self.chains:
                if chain is not chain_to_fold:
                    cmd.delete("chain {}".format(chain))
            self.chains = [chain_to_fold]
        # now get position, velocity, features, masses
        self.step_number = 0
        model = cmd.get_model("current", 1)
        self.original_positions = np.array(model.get_coord_list())
        self.velocities = np.array([[0., 0., 0.] for atom in model.atom])
        self.features = np.array(
            [self.get_atom_features(atom) for atom in model.atom])
        masses = np.array([atom.get_mass() for atom in model.atom])
        self.masses = np.array([masses, masses, masses])

    def get_observation(self):  # chains, aminos, images, atoms, bonds
        observation = {}
        observation["chains"] = self.get_chains()
        observation["amino"] = self.get_amino()
        observation["bond"] = self.get_bonds()
        observation["image"] = self.get_image()
        observation["atom"] = self.get_atom()
        return observation

    def get_chains(self):  # string, num_aminos, num_atoms, x, y, z
        print("mol get_chains")
        print("self.chains", self.chains)
        chains = []
        for chain in self.chains:
            name = ord(chain)/26
            selector = "current and chain {} name ca".format(chain)
            num_aminos = cmd.count_atoms(selector)
            selector = "current and chain {}".format(chain)
            num_atoms = cmd.count_atoms(selector)
            x, y, z = cmd.centerofmass("chain " + chain)
            features = [x, y, z, name, num_aminos, num_atoms]
            chains.append(features)
        print(chains)
        return np.array(chains)

    def get_amino(self):  # chain, resv, oneletter, index, x, y, z
        print("get amino!")
        # get the data
        self.aminos = []
        space = {"aminos": self.aminos}
        cmd.iterate_state(
            -1,
            "(name ca)",
            "aminos.append((chain, resv, oneletter, index, x, y, z))",
            space=space
            )
        # convert the data to numpy array
        aminos = np.array([self.convert_amino(amino) for amino in self.aminos])
        print("aminos", aminos)
        return aminos

    def convert_amino(self, amino):
        chain, resv, oneletter, index, x, y, z = amino
        return np.array([index, x, y, z, ord(amino[0].lower())/26, resv, ord(amino[2].lower())/26])

    def get_bonds(self):
        array = []
        np.array([self.get_bond(array, name, index) for name, index in cmd.index("name ca")])
        return np.array(array)

    def get_bond(self, array, name, index):
        selection_string_array = [
            'first (({}`{}) extend 2 and name C)'.format(name, index),  # prev C
            'first (({}`{}) extend 1 and name N)'.format(name, index),  # this N
            '({}`{})'.format(name, index),                            # this CA
            'last (({}`{}) extend 1 and name C)'.format(name, index),  # this C
            'last (({}`{}) extend 2 and name N)'.format(name, index),  # next N
        ]
        try:
            phi = cmd.get_dihedral(
                selection_string_array[0], selection_string_array[1],
                selection_string_array[2], selection_string_array[3], state=1)
            array.append([phi])
        except:
            print("failed to get dihedral at ", name, index)
        try:
            psi = cmd.get_dihedral(
                selection_string_array[1], selection_string_array[2],
                selection_string_array[3], selection_string_array[4], state=1)
            array.append([psi])
        except:
            print("failed to get dihedral at ", name, index)

    def get_image(self):
        step_indicator = "p{}-{}-s{}".format(
            self.episode,
            self.pdb,
            self.step_number)
        image_paths = self.take_pictures(step_indicator)
        [print(image_path, "\n") for image_path in image_paths]
        images = [np.asarray(Image.open(image)) for image in image_paths]
        [print(image.shape) for image in images]
        vstack = np.vstack(images)
        print("vstack shape", vstack.shape)
        # save the picture
        image_path = self.image_path + "/stacks/" + "{}.jpg".format(
            step_indicator)
        vstack_img = Image.fromarray(vstack)
        vstack_img.save(image_path)
        return vstack

    def take_pictures(self, step_indicator):
        cmd.deselect()
        print("SCREENSHOT!",
              "protein #",
              self.episode,
              "pdb: ",
              self.pdb,
              "step: ",
              self.step_number)
        png_path_x = self.image_path + "/pngs/" + step_indicator + "-X.png"
        cmd.png(png_path_x, width=self.nurture.image_size, height=self.nurture.image_size)
        jpg_path_x = os.path.join(self.image_path, "jpgs", step_indicator + "-X.jpg")
        command = "convert \"{}\" -alpha remove \"{}\"".format(png_path_x, jpg_path_x)
        os.system(command)
        time.sleep(0.02)
        cmd.rotate('y', 90)
        png_path_y = self.image_path + "/pngs/" + step_indicator + "-Y.png"
        cmd.png(png_path_y, width=self.nurture.image_size, height=self.nurture.image_size)
        jpg_path_y = os.path.join(self.image_path, "jpgs", step_indicator + "-Y.jpg")
        command = "convert \"{}\" -alpha remove \"{}\"".format(png_path_y, jpg_path_y)
        os.system(command)
        time.sleep(0.02)
        cmd.rotate('y', -90)
        cmd.rotate('z', 90)
        png_path_z = self.image_path + "/pngs/" + step_indicator + "-Z.png"
        cmd.png(png_path_z, width=self.nurture.image_size, height=self.nurture.image_size)
        jpg_path_z = os.path.join(self.image_path, "jpgs", step_indicator + "-Z.jpg")
        command = "convert \"{}\" -alpha remove \"{}\"".format(png_path_z, jpg_path_z)
        os.system(command)
        time.sleep(0.02)
        cmd.rotate('z', -90)
        return (jpg_path_x, jpg_path_y, jpg_path_z)

    def get_atom(self):
        model = cmd.get_model("current", 1)
        self.positions = np.array(model.get_coord_list())
        print("positions", self.positions.shape)
        print("velocities", self.velocities.shape)
        print("features", self.features.shape)
        return np.column_stack((
            self.positions,
            self.velocities,
            self.features
            ))

    def get_atom_features(self, atom):
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

    def load_pedagogy(self):
        results = []
        csvpath = os.path.join(self.task_root, "inputs/csvs/2chains.csv")
        with open(csvpath) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                for item in row:
                    item = item.strip()
                    results.append(item)
        return results

    def prepare_pymol(self):
        cmd.remove("solvent")
        self.color_chainbow()
        cmd.set("depth_cue", 0)
        cmd.unpick()
        cmd.show('surface')
        cmd.zoom("all", buffer=42, state=-1)
        cmd.center("all")
        cmd.bg_color("black")
        cmd.deselect()

    def color_chainbow(self):
        for chain in self.chains:
            chain_string = "chain " + chain
            color = self.pick_a_color()
            print("coloring", chain_string, color)
            cmd.color(color, chain_string)

    def pick_a_color(self):
        colors = ["red", "orange", "yellow", "green", "blue", "marine", "violet"]
        color_is_chosen = False
        while not color_is_chosen:
            color = random.choice(colors)
            if color not in self.colors_chosen:
                self.colors_chosen.append(color)
                color_is_chosen = True
                return color

    def make_gif(self):
        gif_name = "r{}-p{}-{}.gif".format(
            self.run_time_stamp,
            self.episode,
            self.pdb)
        gif_path = os.path.join(self.root, "results/gifs", gif_name)
        imagepaths = []
        images = []
        for stackname in os.listdir(self.image_path + "/stacks"):
            filepath = os.path.join(self.image_path, "stacks", stackname)
            imagepaths.append(filepath)
        imagepaths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for imagepath in imagepaths:
            print("processing ", imagepath)
            image = imageio.imread(imagepath)
            images.append(image)
        print("saving gif to", gif_path)
        imageio.mimsave(gif_path, images)
        return gif_path, gif_name

    def slack_file(self, file_path, file_name):
        slack = SlackClient(self.nurture.slack_token)
        slack.api_call('files.upload',
                       channels='#gifs',
                       as_user=True,
                       filename=file_name,
                       file=open(file_path, 'rb'),)

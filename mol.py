
from PIL import Image
import numpy as np
import imageio
import random
import shutil
import scipy
import time
import gym
import csv
import os

from spaces import Array
import pymol

cmd = pymol.cmd

class PyMolEnv(gym.Env):
    def __init__(self, config):
        # setup
        self.run_time_stamp = str(time.time())
        self.root = os.path.dirname(os.path.abspath("."))
        self.images_path = os.path.join(self.root, "images")
        self.csvs_path = os.path.join(self.root, "csvs")
        self.pdbs_path = os.path.join(self.root, "pdbs")
        self.gifs_path = os.path.join(self.root, "gifs")
        self.pedagogy = self.load_pedagogy()
        self.colors_chosen = []
        self.config = config
        self.episode = 0
        self.episode_images_path = './images'
        # define spaces
        self.observation_space = Array(shape=(None, 17))
        self.action_space = Array(shape=(None, 3))

    def reset(self):
        self.episode += 1
        # clean up!
        cmd.delete("all")
        # set up image path
        self.episode_images_path = os.path.join(self.images_path, self.run_time_stamp, str(self.episode))
        os.makedirs(self.episode_image_paths)
        self.episode_stacks_path = os.path.join(self.episode_image_paths, "stacks")
        os.makedirs(self.episode_stacks_path)
        self.episode_stacks_path = os.path.join(self.episode_image_paths, "pngs")
        # load a pdb
        self.pdb = random.choice(self.pedagogy) + ".pdb"
        self.pdb_path = os.path.join(self.pdbs_path, self.pdb)
        print("loading", self.pdb_path)
        if not os.path.exists(self.pdb_path):
            cmd.fetch(self.pdb, path=self.pdbs_path, type="pdb")
        elif os.path.exists(self.pdb_path):
            cmd.load(self.pdb_path)
        cmd.remove("solvent")
        # 5. prepare for task setup
        cmd.select(name="current", selection="all")
        self.get_metadata()
        self.undock()
        # update action space with the shape
        self.action_space = Array(
            shape=self.velocities.shape,
            high=4.2,
            low=-4.2
            )
        self.current = self.get_observation()
        # we need to know the initial work to calculate stop loss
        self.initial_work = self.calculate_reward()
        return self.current

    def undock(self):
        print("mol undock chains", self.chains)
        steps_in_undock = random.randint(
            self.config.MIN_STEPS_IN_UNDOCK,
            self.config.MAX_STEPS_IN_UNDOCK
            )
        print("undocking", self.pdb, steps_in_undock, "times")
        sum_vector = {}
        step_vector_array = []
        min_undock = self.config.MIN_UNDOCK_DISTANCE
        max_undock = self.config.MAX_UNDOCK_DISTANCE
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

    # we execute physics here:
    def step(self, action):
        force = -1 * action["potentials"]
        acceleration = force / np.transpose(self.masses)
        noise = np.random.normal(
            loc=0.0,
            scale=self.config.ATOM_JIGGLE,
            size=self.velocities.shape)
        print("acceleration dtype", acceleration.dtype, "noise dtype", noise.dtype)
        self.velocities += acceleration + noise
        self.atom_index = 0
        np.array([self.move_atom(xyz) for xyz in self.velocities])
        self.positions += self.velocities
        reward = self.calculate_reward()
        stop_loss = self.work > self.initial_work * self.config.STOP_LOSS_MULTIPLE
        beyond_max_steps = self.step_number > self.config.NUM_STEPS
        done = stop_loss or beyond_max_steps
        observation = self.get_observation()
        info = None
        self.step_number += 1
        if done:
            print("done because beyond_max_steps", beyond_max_steps)
            print("done because stop loss", stop_loss)
            self.make_gif()
            # delete the image folder to save space
            shutil.rmtree(self.episode_images_path)
        return observation, reward, done, info
    # we move 1 atom
    def move_atom(self, vector):
        movement_vector = list(vector)
        atom_selection_string = "id " + str(self.atom_index)
        cmd.translate(movement_vector, atom_selection_string)
        self.atom_index += 1
    # we calculate reward
    def calculate_reward(self):
        distance = scipy.spatial.distance.cdist(
            self.positions,
            self.original_positions
            )
        # calculate work to move the atoms to the proper spot
        mass = np.transpose(self.masses[0,:])
        self.work = distance * mass
        self.work = np.triu(self.work, 1)  # k=1 should zero main diagonal
        self.work = np.sum(self.work)
        # calculate work to slow the atoms to a stop
        work_to_stop = np.transpose(self.velocities) * mass
        work_to_stop = np.triu(work_to_stop, 1)
        work_to_stop = np.sum(self.work)
        if work_to_stop > 0:
            self.work += work_to_stop
        # reward is the opposite of work
        return -1 * self.work

    def get_metadata(self):
        self.chains = cmd.get_chains("current")
        self.step_number = 0
        model = cmd.get_model("current", 1)
        self.original_positions = np.array(model.get_coord_list())
        self.velocities = np.array([[0., 0., 0.] for atom in model.atom])
        self.features = np.array(
            [self.get_atom_features(atom) for atom in model.atom])
        masses = np.array([atom.get_mass() for atom in model.atom])
        self.masses = np.array([masses, masses, masses])

    def get_observation(self):  # chains, aminos, images, atoms, bonds
        observation = self.get_atom()
        return observation

    def get_image(self):
        step_indicator = "e-{}_p-{}_s-{}".format(
            self.episode,
            self.pdb,
            self.step_number)
        image_paths = self.take_pictures(step_indicator)
        # [print(image_path, "\n") for image_path in image_paths]
        images = [np.asarray(Image.open(image)) for image in image_paths]
        # [print(image.shape) for image in images]
        vstack = np.vstack(images)
        # print("vstack shape", vstack.shape)
        # save the picture
        image_path = os.path.join(self.episode_img_paths, "stacks", step_indicator + ".png")
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
        png_path_x = os.path.join(self.episode_images_path, "pngs", step_indicator + "-X.png")
        cmd.png(png_path_x, width=self.config.IMAGE_SIZE, height=self.config.IMAGE_SIZE)
        time.sleep(0.02)
        cmd.rotate('y', 90)
        png_path_y = os.path.join(self.episode_images_path, "pngs", step_indicator + "-Y.png")
        cmd.png(png_path_y, width=self.config.IMAGE_SIZE, height=self.config.IMAGE_SIZE)
        time.sleep(0.02)
        cmd.rotate('y', -90)
        cmd.rotate('z', 90)
        png_path_z = os.path.join(self.episode_images_path + "pngs" + step_indicator + "-Z.png")
        cmd.png(png_path_z, width=self.config.IMAGE_SIZE, height=self.config.IMAGE_SIZE)
        time.sleep(0.02)
        cmd.rotate('z', -90)
        return (png_path_x, png_path_y, png_path_z)

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
        csvpath = os.path.join(self.root, "neuromax-2019/csvs/pedagogy.csv")
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
        gif_name = "r-{}_e-{}_p-{}.gif".format(
            self.run_time_stamp,
            self.episode,
            self.pdb)
        gif_path = os.path.join(self.root, "gifs", gif_name)
        imagepaths = []
        images = []
        for stackname in os.listdir(self.episode_stacks_path):
            filepath = os.path.join(self.episode_stacks_path, stackname)
            imagepaths.append(filepath)
        imagepaths.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for imagepath in imagepaths:
            print("processing ", imagepath)
            image = imageio.imread(imagepath)
            images.append(image)
        print("saving gif to", gif_path)
        imageio.mimsave(gif_path, images)
        return gif_path, gif_name

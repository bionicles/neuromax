# neuromax-pymol-tensorflow.py - bion and kamel - 10 august 2018
# why?: model-free ( automatic ) molecular dynamics
# how?: utils, Searcher, Input_Loader, Model, Trainer
# what?: desired workflow -> run pymol -qc neuromax-pymol-tensorflow.py | tensorboard --logdir=./results
# screenshots + gifs start to appear in the "results/{TIMESTAMP}

import itertools, imageio, random, pymol, time, os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pymol import cmd, util

class AttrDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__

def prepare_pymol():
    cmd.set("ray_trace_color", "black")
    cmd.set("ray_trace_gain", 0)
    cmd.set("ray_trace_mode", 3)
    cmd.set("depth_cue", 0)
    cmd.clip("near", 100000)
    cmd.clip("slab", 100000)
    cmd.clip("far", -100000)
    cmd.hide("all")
    util.cbc()
    cmd.show('surface')
    cmd.bg_color("black")
    cmd.zoom("all", buffer=2.0) # buffer might help clipping problem
    cmd.center("all")
    cmd.deselect()

def take_screenshot(params, pdb_file_name, protein_number, step_number):
    print("")
    print("SCREENSHOT! protein #", protein_number, pdb_file_name, " step ", step_number, " with ", cmd.count_atoms("all"), " atoms")
    screenshot_path = "./results/{}/screenshots/{}-{}/{}-{}.png".format(params.run_time_stamp, pdb_file_name, protein_number, pdb_file_name, step_number)
    cmd.ray()
    # cmd.zoom("all")
    cmd.center("all")
    cmd.png(screenshot_path)

def make_gif(params, pdb_file_name, protein_number):
    print("begin annotating screenshots")
    ss_dir = "./results/{}/screenshots/{}-{}/".format(params.run_time_stamp, pdb_file_name, protein_number)
    original_image_path = ss_dir + pdb_file_name + "-0.png"
    for f in os.listdir(ss_dir):
        file_name,file_extension=os.path.splitext(f)
        if file_name[-2:] == "-0":
            print("skipping original name", original_image_path)
            continue
        print(file_name,ss_dir+file_name+file_extension+'***************')
        images_list = [ss_dir + pdb_file_name + '-0.png', ss_dir + file_name + file_extension]
        print(images_list)
        imgs = [Image.open(i) for i in images_list]
        min_img_shape = sorted([(np.sum(i.size), i.size ) for i in imgs])[0][1]
        img = np.hstack((np.asarray( i.resize(min_img_shape,Image.ANTIALIAS)) for i in imgs ) )
        img = Image.fromarray(img)
        x,y = img.size
        draw = ImageDraw.Draw(img)
        #font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-B.ttf", size=32)
        pdb_id_string = "PDB: " + pdb_file_name
        x2, y2 = draw.textsize(pdb_id_string)
        model_id_string = "Bit Pharma Neuromax: " + params.run_time_stamp
        x3, y3 = draw.textsize(model_id_string)
        draw.text(((x-x2)/2, 10), pdb_id_string,(255,255,255))
        draw.text(((x-x3)/2, y-42),model_id_string,(255,255,255))
        img.save(ss_dir+file_name+'.png')
    print("done annotating screenshots")
    print("begin generating gif")
    dirfiles = []
    images = []
    for filename in os.listdir(ss_dir):
        if filename[-3:] == "png":
            dirfiles.append(filename)
    dirfiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for dirfile in dirfiles[1:]:
        filepath = os.path.join(ss_dir, dirfile)
        print("processing ", filepath)
        images.append(imageio.imread(filepath))
    gif_dir = "./results/{}/gifs".format(params.run_time_stamp)
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
    gif_name = "neuromax." + params.run_time_stamp + "." + pdb_file_name + "-" + str(protein_number) + ".gif"
    gif_path = os.path.join(gif_dir, gif_name)
    imageio.mimsave(gif_path, images)
    print("done generating gif at ", gif_path)
    return(gif_path)

class Searcher(object): # hyperparameter searcher
    def __init__(self):
        super(Searcher, self).__init__()

    def search(self): # create hyperparameter dictionaries # TODO: test more random stuff!
        params = AttrDict()
        run_time_stamp = str(time.time())
        params.run_time_stamp = run_time_stamp
        params.log_folder_path = "./results/{}/".format(run_time_stamp)
        params.screenshot_folder_path = params.log_folder_path + "screenshots/"
        os.makedirs(params.screenshot_folder_path)
        
        # loss parameters
        params.min_center_distance_loss_weight = 0.0
        params.max_center_distance_loss_weight = 100
        params.min_meta_distance_loss_weight = 1
        params.max_meta_distance_loss_weight = 1
        params.min_mean_distance_loss_weight = 0
        params.max_mean_distance_loss_weight = 0
        # simulation parameters
        params.initial_vector_divisor = random.randint(42,100)
        params.noise_scale = random.uniform(0.0027, 0.0081) # TODO: add solvent
        params.noise_mean = random.uniform(0.01, 0.02)
        params.max_steps_in_undock = 3
        params.min_steps_in_undock = 0
        params.max_translation = 42
        params.min_translation = 1
        params.initial_steps_to_screenshot = 0
        params.screenshot_every = 100
        params.learning_rate = random.uniform(0.001,0.0042)
        params.max_num_epochs =  9
        params.min_num_epochs = 1
        params.use_pedagogy = False
        params.steps_in_gif = 42
        params.num_epochs = 1
        params.input_dim = 15
        # neural net parameters
        params.ACTIVATION=  "tanh"
        params.NUM_BLOCKS = 6
        params.NUM_LAYERS = 2
        params.UNITS = 50
        params.DROPOUT = False
        params.DROPOUT_RATE = 0.1
        params.CONNECT_BLOCK_AT = 2
        params.OUTPUT_SHAPE = 366
        params.LOSS_FUNCTION = "mse"
        params.OPTIMIZER = "adam"
        return params

class Input_Loader(object): # https://www.tensorflow.org/guide/datasets
    def __init__(self, params):
        super(Input_Loader, self).__init__()
        self.dataset = tf.data.Dataset.from_generator(self.load_pdb, tf.float32, tf.TensorShape([None, 3])).repeat()
        self.iterator = self.dataset.make_one_shot_iterator()
        self.params = params
        self.pedagogy = ['4unh', '1bzv', '4i5y', '5d52', '1b2g', '4iuz', '5d5e', '1b19', '1b2e', '2g4m', '1b18', '1b2f', '1b2c', '1b2b', '1b2a', '1cph', '1b2d', '1b17', '5usp', '1aph', '9ins', '1ncp', '4iyf', '1ioh', '1mhj', '2n2v', '4oh8', '2mvc', '2vpb', '3e7z', '2kxk', '4ung', '5h8o', '2c7m', '6gb1', '3c59', '3dvj', '1b0n', '1xou', '1b34', '4nut', '3dvk', '5lz3', '1wa7', '2hqw', '4x8k', '1iq5', '2nla', '3fpu', '4k12', '7ins', '4m1l', '5j26', '2o2v', '5fcg', '1tfo', '1a2x', '4c4k', '1m45', '1e44', '1wrd', '1ay7', '1ben', '4m4l', '2l1r', '2wpt', '1a7q', '3fxd', '1ykh', '5gwm', '4lrn', '5nrm', '3oss', '1b8m', '1y01', '2wwk', '3rnk', '3znh', '5adq', '2mbb', '4dzv', '1ct0', '2sgq', '1fr2', '1wkw', '3fap', '1a6u', '3vvw', '2m8s', '1mfa', '5t86', '1bdj', '5b5w', '2a7u', '2p44', '5xbo', '1k8r', '6amb', '1cfv', '4a5u', '2sta', '2vln', '4lci', '1an1', '2lvo', '4zk9', '1zvh', '4dri', '5obz', '5vgb', '5vkl', '2n01', '1ri8', '3ogi', '4w6x', '3kf6', '2g2s', '3bte', '3btg', '1brc', '2p49', '2mfq', '2tpi', '5otx', '1t0p', '2n9p', '1tgs', '2k6d', '2hsq', '5dfl', '2plx', '3tpi', '1nme', '4tpi', '1iql', '3sw2', '1g2m', '4j2l', '3bog', '1fy8', '4ayf', '1v3x', '1ylc', '5ywr', '1p2k', '4zdt', '1aks', '2q1j', '5gpy', '2thf', '1syq', '3q87', '1acb', '2p3t', '1hcg', '2j4i', '3kl6', '2j34', '2li5', '4euk', '3m37', '5vmo', '4i1s', '2anm', '1abj', '3tj5', '3ddc', '1wmi', '3cen', '2hle', '4rkj', '1tq7', '5odt', '1z8j', '2wo2', '1ksg', '5n7e', '4iso', '1tgz', '5d6h', '3bk3', '3fp6', '4zxy', '3uzv', '2zup', '3dxe', '4jyv', '2cg5', '4iyp', '5u6j', '3vuy', '2co6', '3pk1', '3reb', '2l1w', '5tp6', '1iar', '2grn', '4jzd', '4odc', '3n7r', '4dex', '4jze', '5pb0', '1cse', '5pa8', '2v9t', '5pb2', '3cbk', '2vof', '5l2y', '2i32', '5pau', '5oec', '5pas', '1qfk', '3cup', '1l4z', '5l0w', '1jlu', '3lqc', '1ydr', '2mma', '3zld', '5fr2', '4uem', '1as4', '1ur6', '2aq2', '1avx', '1avw', '2nxn', '3d48', '2doi', '1atp', '1apm', '5oe7', '3amb', '2z57', '5e1h', '3cbj', '4bru', '1a22', '4bl7', '1tmg', '1y3f', '5uwc', '1htr', '4zgn', '4nc0', '3pc8', '1y33', '4fbj', '4x6e', '5f1o', '4x6f', '1grn', '3msx', '5c2j', '4xsh', '2gol', '1smh', '4beh', '4iaz', '5l8k', '1bey', '2aw2', '4iad', '1a5f', '5c9j', '5waw', '2uw0', '2wy8', '1ay1', '2fgw', '8api', '1a0q', '1q8u', '1usu', '4dfz', '2c0l', '5om5', '4yxs', '1bbd', '5whj', '4n6o', '2vnw', '3mp7', '4nki', '4dg0', '1lo4', '5ukp', '5d7s', '1rih', '2xn6', '5alc', '6blx', '2dtm', '2b0z', '3rea', '1for', '4ww7', '2vo6', '1i9i', '1etr', '6blr', '5v52', '2h4q', '5v90', '3gmq', '2cdg', '4g01', '1zli', '2qxv', '3ul4', '4hxb', '3mzg', '2o5y', '5ukq', '4dka', '1cr9', '5c1m', '4nja', '5e2t', '5hvq', '1riv', '5wkg', '6b0w', '3qnx', '1m7i', '1riu', '4yjz', '4kbq', '4isv', '1yeg', '3sbw', '1ggp', '3fju', '1f8t', '3e90', '2f5a', '1axt', '1dqm', '3ga9', '3ia3', '4l2f', '6fg8', '2jjt', '1a3l', '6blq', '1wt5', '1ets', '1q9k', '3x24', '6bwa', '1s3k', '4hlm', '4u3x', '1aut', '3r24', '1ap2', '5yy0', '1ugq', '2r1y', '2p46', '2zph', '4l32', '1ugs', '4jm4', '4hlf', '3uvv', '2r1x', '1znv', '1loe', '1ira', '3vhl', '3ukx', '5od0', '5xns', '3mcl', '1ugr', '1r20', '3okk', '1s1q', '3bh6', '3hnv', '5ixi', '3woe', '3o43', '5o85', '1fq1', '3t77', '5n4j', '4lga', '5vsq', '3c5x', '2qwq', '1te1', '5nwb', '1psk', '4pwv', '4ui6', '3wcy', '5wuv', '2bpb', '4ww1', '4gs9', '2r40', '3if8', '4udt', '1or7', '1r6q', '5e1a', '1aj7', '5aqu', '6fuo', '1baf', '2cz7', '1ijf', '1hwo', '3v6i', '3bdy', '2qwo', '4ott', '4w4k', '1nmd', '5aqv', '4crv', '5mu7', '5kp7', '5c6p', '3zkq', '5ms2', '3n9g', '1lvm', '3d1m', '5a32', '1d4x', '2w2x', '2ckz', '2rg9', '1ny7', '3o5w', '1t44', '3aa7', '5ml9', '2w9z', '4msq', '1xzq', '5e6p', '1pum', '4kt1', '4zc6', '1smp', '4msm', '4apx', '3hf4', '1efv', '3cip', '6fbz', '4bci', '5da7', '3whq', '5lxq', '3lpe', '1y14', '3byh', '5eb1', '1aui', '1p2m', '1awh', '4z0k', '5lgd', '5lx9', '1b41', '1b3s', '4zri', '3cr3', '1s70', '3dut', '1atn', '5xlu', '3fgr', '1b27', '1b2u', '1o6s', '3rgf', '4crl', '3c6n', '4tu3', '4o9p', '4bti', '4b2c', '5hbe', '1x1x', '3cpj', '3f1s', '5tgj', '4rsi', '4gts', '4zjv', '4f7s', '3gni', '1b2s', '4nko', '5jjd', '4din', '3oad', '3ens', '5oen', '1tu2', '4gtv', '1euc', '2qcs', '5wxl', '2omv', '1i1q', '5lxy', '4v0v', '3d3c', '5w83', '4u6u', '1rgi', '5x5w', '4x6r', '5jfz', '4cj0', '4s0s', '1miu', '4o4b', '1gaq', '1avf', '3dss', '1ewy', '1shy', '3zdf', '2iyb', '5oed', '2znx', '1abr', '3onw', '3src', '4w2q', '1b5f', '1q40', '5vau', '4ydo', '4o8x', '1a0o', '3tho', '5d3i', '2o8g', '2okr', '3eu5', '2vgp', '1pvh', '2qna', '4c2w', '1a6z', '1lpb', '3mwd', '4qxf', '4lhq', '1ak4', '4uec', '4nzu', '4fqx', '3pz4', '4f7c', '2wye', '3e33', '4kph', '2wpn', '2zis', '4zxs', '5tl6', '4czz', '3zrk', '4ni2', '1cdk', '2beh', '5lgu', '1k7d', '1uug', '5lbq', '1pnm', '2iw5', '4y8d', '3e30', '5om6', '1r24', '1bbj', '1a6t', '5ky4', '6gbh', '5h24', '3rgv', '4a6y', '1ai6', '4h0v', '5dtf', '5m6v', '3oed', '1gzh', '2p8w', '5wcd', '2ntf', '1aif', '1sa4', '1ai4', '1fve', '4rir', '2gk0', '2gjz', '3e37', '5kxh', '1ai7', '1ad0', '1ai5', '1gm8', '1ajn', '5vzx', '3nz8', '1ajp', '1ajq', '3gfu', '4llw', '6ei1', '5vz1', '5nvn', '1mf2', '3ijy', '1axs', '4nrz', '1s0w', '1a4k', '5m63', '2a1w', '1nc4', '3k7w', '1u74', '5uxq', '5e5m', '4g6v', '4odu', '3if1', '3tv3']
        # self.pdb_file_names = [x.strip(' ') for x in self.pdb_file_names]
        # self.pdb_file_names = pd.read_csv(self.params.pdb_list_csv_file_path).columns.values

    def load_pdb(self, num_proteins_seen, screenshotting, pdb_file_name=""): # use pymol wiki
        cmd.delete("all") # prevent memory issues
        # choose a pdb id
        if num_proteins_seen > len(self.pedagogy):
            self.pdb_file_name = random.sample(self.pedagogy, 1)[0]
        elif (num_proteins_seen < len(self.pedagogy) and pdb_file_name == ""):
            self.pdb_file_name = self.pedagogy[num_proteins_seen]
        # fetch or load pdb
        self.pdb_file_path = "./pdbs/"+self.pdb_file_name+".pdb"
        if not os.path.exists(self.pdb_file_path):
            cmd.fetch(self.pdb_file_name, path="./inputs/pdbs", type="pdb")
        elif os.path.exists(self.pdb_file_path):
            cmd.load(self.pdb_file_path)
        cmd.remove("solvent")
        # summarize
        print(self.params.run_time_stamp, " is loading ", num_proteins_seen, self.pdb_file_path)
        print("")
        num_atoms = cmd.count_atoms("all")
        print("noise mean", self.params.noise_mean, "noise scale", self.params.noise_scale)
        # convert pdb2tensor
        original_model = cmd.get_model('all', 1)
        original_coords_list = cmd.get_model('all', 1).get_coord_list()
        original = tf.convert_to_tensor(np.array(original_coords_list), dtype=tf.float32)
        chains = cmd.get_chains()
        if (screenshotting):
            self.current_pdb_screenshot_path = self.params.screenshot_folder_path + "/" + self.pdb_file_name + "-" + str(num_proteins_seen) + "/"
            os.makedirs(self.current_pdb_screenshot_path)
            prepare_pymol()
            take_screenshot(self.params, self.pdb_file_name, num_proteins_seen, "0")
        num_steps = random.randint(self.params.min_steps_in_undock, self.params.max_steps_in_undock)
        self.undock(num_proteins_seen, screenshotting, num_steps, chains)
        undocked_coords_list = cmd.get_model('all', 1).get_coord_list()
        undocked = tf.convert_to_tensor(np.array(undocked_coords_list), dtype=tf.float32)
        # calculate center of mass dict
        self.center_of_mass_dict = AttrDict()
        self.center_of_mass_dict["all"] = cmd.centerofmass("all")
        for chain in chains:
            self.center_of_mass_dict[chain] = cmd.centerofmass("chain {}".format(chain))
        features = np.array([self.extract(atom) for atom in original_model.atom])
        features = tf.convert_to_tensor(features, dtype=tf.float32)
        #outputs
        output_tuple = (self.center_of_mass_dict, num_steps, undocked, features, original, chains)
        return output_tuple

    # {'chain': 'A', 'resi': '2', 'resn': 'ASP', 'formal_charge': 0, 'id': 1, 'numeric_type': 0, 'index': 1, 'resi_number': 2, 'symbol': 'N', 'u_aniso': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'elec_radius': 0.0, 'stereo': 0, 'b': 53.4900016784668, 'name': 'N', 'vdw': 1.5499999523162842, 'ss': '', 'coord': [28.663000106811523, -3.9800000190734863, 6.677000045776367], 'hetatm': 0, 'q': 1.0, 'flags': 134217728, 'partial_charge': 0.0, 'segi': ''}
    # [dx, dy, dz, mass, charge, vdw, b, free_valence, resi#, chain, resn, symbol]
    def extract(self, atom):
        dx, dy, dz = (-1 * np.array(self.center_of_mass_dict["all"]) - np.array(self.center_of_mass_dict[atom.chain]) / self.params.initial_vector_divisor) + np.random.normal(self.params.noise_mean, self.params.noise_scale, 3)
        features = np.array([dx, dy, dz, atom.get_mass(), atom.formal_charge, atom.vdw, atom.b, atom.get_free_valence(0), sum([ord(i) for i in atom.resi])//len(atom.resi), ord(atom.chain), sum([ord(i) for i in atom.resn])//len(atom.resn), sum([ord(i) for i in atom.symbol])//len(atom.symbol) ])
        return features

    def undock(self, num_proteins_seen, screenshotting, num_steps, chains):
        sum_vector = {}
        steps = []
        # make a list of steps
        for step in range(num_steps):
            current_step_vectors = []
            for chain in chains:
                # x, y, z, rx, ry, rz
                vector = np.array([random.randint(self.params.min_translation, self.params.max_translation),
                random.randint(self.params.min_translation, self.params.max_translation),
                random.randint(self.params.min_translation, self.params.max_translation),
                random.randint(-360, 360),
                random.randint(-360, 360),
                random.randint(-360, 360)])
                # add them up (final destination of undock)
                if chain not in sum_vector.keys():
                    sum_vector[chain] = vector
                else:
                    sum_vector[chain] = sum_vector[chain] + vector
                current_step_vectors.append(vector)
            steps.append(current_step_vectors)
        # move to the final position
        for chain in chains:
            if chain in sum_vector.keys():
                vector = sum_vector[chain]
                final_translation_vector = list(vector[:3])
                cmd.translate(final_translation_vector, "chain " + chain)
                cmd.rotate("x", vector[3], "chain " + chain)
                cmd.rotate("y", vector[4], "chain " + chain)
                cmd.rotate("z", vector[5], "chain " + chain)
        # if screenshotting this simulation, animate the steps
        if (screenshotting):
            step_number = 0
            for chain in chains:
                if chain in sum_vector.keys():
                    vector = sum_vector[chain] * -1 # move back
                    inverse_translation_vector = list(vector[:3])
                    cmd.translate(inverse_translation_vector, "chain " + chain)
                    cmd.rotate("x", vector[3], "chain " + chain)
                    cmd.rotate("y", vector[4], "chain " + chain)
                    cmd.rotate("z", vector[5], "chain " + chain)
                prepare_pymol() # set the zoom on the final destination
                take_screenshot(params, self.pdb_file_name, num_proteins_seen, str(step_number + 1))
                step_number = step_number + 1
            # move through the list of steps, move the chains, take the screenshots
            for vectors in steps:
                for k in range(len(vectors)):
                    vector = vectors[k]
                    chain = chains[k]
                    current_translation_vector = list(vector[:3])
                    cmd.translate(current_translation_vector, "chain " + chain)
                    cmd.rotate("x", vector[3], "chain " + chain)
                    cmd.rotate("y", vector[4], "chain " + chain)
                    cmd.rotate("z", vector[5], "chain " + chain)
                take_screenshot(params, self.pdb_file_name, num_proteins_seen, str(step_number + 1))
                step_number = step_number + 1

class resnet(tf.keras.Model):
    def __init__ (self, params):
        super(resnet, self).__init__(name='resnet')
        self.params = params

    def make_block(self, input_layer, params, block_num):
        print(self.params.input_dim, "*********************")
        output = tf.keras.layers.Dense(units = params.UNITS, activation=params.ACTIVATION)(input_layer)
        for layer_num in range(params.NUM_LAYERS - 1):
            output = tf.keras.layers.Dense(units = params.UNITS, activation=params.ACTIVATION, name = 'block_{}'.format(block_num))(output)
            if params.DROPOUT: #True or False
                output = tf.keras.layers.Dropout(rate = params.DROPOUT_RATE)(output) # not sure if dropout layer support names option
        return output

    def make_model(self):
        input = tf.keras.layers.Input(shape=(params.input_dim, ))
        output = self.make_block(input, params, block_num = 0) # initial block
        output_shortcut = output
        for i in range(1, params.NUM_BLOCKS): #start count blocks from 1, useful to connect blocks using their numbers
            output = self.make_block(output, params, i)
            if i%params.CONNECT_BLOCK_AT == 0:
                output = tf.keras.layers.average([output, output_shortcut])
                output_shortcut = output
        model_output = tf.keras.layers.Dense(units = params.OUTPUT_SHAPE, activation = params.ACTIVATION)(output) 
        model = tf.keras.models.Model(input, model_output)
        model.compile(loss = params.LOSS_FUNCTION, optimizer = params.OPTIMIZER)
        tf.keras.utils.plot_model(model, show_shapes = True)
        return model

class Trainer(object):
    def __init__(self, params, searcher, input_loader):
        super(Trainer, self).__init__()
        self.params = params
        self.searcher = searcher
        self.input_loader = input_loader
        self.steps_to_screenshot = self.params.initial_steps_to_screenshot
        self.screenshotting = False
        self.model = resnet(self.params).make_model()
        self.model.summary()
        #self.adam = tf.train.AdamOptimizer() # https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer#
        #self.model.compile(loss=self.loss, optimizer='adam')
        self.checkpoint_dir = './results/{}/checkpoints'.format(self.params.run_time_stamp)
        self.checkpoint_prefix = self.checkpoint_dir + "/ckpt"
        self.scaler = MinMaxScaler()
        #self.saver = tfe.Checkpoint(optimizer=self.adam, model=self.model, optimizer_step=tf.train.get_or_create_global_step())

    def train(self):
        self.num_proteins_seen = 1
        self.delta_loss=[0]
        self.list=[]
        while True:
            # decide about screenshots
            print("")
            if (self.steps_to_screenshot == 0):
                self.params.num_epochs = self.params.steps_in_gif
                self.screenshotting = True
                self.steps_to_screenshot = self.params.screenshot_every
            else:
                self.params.num_epochs = random.randint(self.params.min_num_epochs, self.params.max_num_epochs)
                print(self.steps_to_screenshot, "steps to screenshot")
                self.steps_to_screenshot = self.steps_to_screenshot - 1
            # get a new protein
            (self.center_of_mass_dict, num_steps, self.undocked, self.features, self.original, self.chains) = self.input_loader.load_pdb(num_proteins_seen=self.num_proteins_seen, screenshotting=self.screenshotting)
            self.features = self.features.numpy()
            self.mass_array = self.features[:, [3]]
            self.mass_tensor =  tf.convert_to_tensor(self.mass_array, dtype = tf.float32)
            self.num_steps_taken = 1 + num_steps
            self.velocity = self.features[:, [0,1,2]]
            self.original_distances = self.calculate_distance_tensor(self.original)
            self.center_of_mass = tf.convert_to_tensor(np.array(self.input_loader.center_of_mass_dict["all"]), dtype=tf.float32)
            # fit model and plot results
            x = self.scaler.fit_transform(np.concatenate((self.undocked, self.features), 1), self.original_distances)
            print("x shape: ", x.shape)
            print("y shape: ", self.original_distances.shape)
            self.history=self.model.fit(x=x, y=self.original_distances, epochs=self.params.num_epochs, batch_size=self.undocked.numpy().shape[0], verbose=1)
            self.list.extend(self.history.history['loss'])
            self.plot()
            # save gif, model
            if self.screenshotting:
                make_gif(self.params, self.input_loader.pdb_file_name, self.num_proteins_seen)
                self.save()
                self.model.summary()
                self.screenshotting = False
            # iterate
            self.num_proteins_seen = self.num_proteins_seen + 1

    def calculate_distance_tensor(self, position_tensor):
        # print("calculate_distance_tensor", position_tensor.shape)
        distances = tf.reduce_sum(position_tensor * position_tensor, 1)
        distances = tf.reshape(distances, [-1,1])
        distances = distances - 2 * tf.matmul(position_tensor, tf.transpose(position_tensor)) + tf.transpose(distances)
        # print("distance tensor shape", distances.shape)
        return distances

    def loss(self, original, predicted_forces):
        # print("loss", original, predicted_forces)
        self.acceleration_tensor = predicted_forces / self.mass_tensor
        new_noise = tf.convert_to_tensor(np.random.normal(self.params.noise_mean, self.params.noise_scale, self.velocity.shape), dtype=tf.float32)
        self.velocity = self.velocity + self.acceleration_tensor + new_noise
        # if screenshotting move and screenshot
        if (self.screenshotting):
            self.translate() # TODO: test move this into if (self.screenshotting) to save compute time
            take_screenshot(self.params, self.input_loader.pdb_file_name, self.num_proteins_seen, self.num_steps_taken + 1)
        self.undocked = self.undocked + self.velocity
        self.undocked_distances = self.calculate_distance_tensor(self.undocked)
        # return weighted sum of losses
        center_distance_loss_weight = random.uniform(self.params.min_center_distance_loss_weight, self.params.max_center_distance_loss_weight)
        chain_number = 0
        for chain in self.chains:
            if chain_number == 0:
                center_distance_loss_tensor = tf.losses.absolute_difference(np.array(self.center_of_mass_dict['all']), np.array(self.center_of_mass_dict[chain]))
            else:
                current_chain_center_distance_loss = tf.losses.absolute_difference(np.array(self.center_of_mass_dict['all']), np.array(self.center_of_mass_dict[chain]))
                center_distance_loss_tensor = center_distance_loss_tensor + current_chain_center_distance_loss
            chain_number = chain_number + 1
        meta_distance_loss_weight = random.uniform(self.params.min_meta_distance_loss_weight, self.params.max_meta_distance_loss_weight)
        mean_distance_loss_weight = random.uniform(self.params.min_meta_distance_loss_weight, self.params.max_meta_distance_loss_weight)
        self.meta_distance_loss_tensor = tf.losses.absolute_difference(self.original_distances, self.undocked_distances)
        self.mean_distance_loss_tensor = tf.losses.mean_squared_error(self.original_distances, self.undocked_distances)
        self.weighted_sum_of_loss_tensors = meta_distance_loss_weight * self.meta_distance_loss_tensor + mean_distance_loss_weight * self.mean_distance_loss_tensor + center_distance_loss_weight * center_distance_loss_tensor
        self.num_steps_taken = self.num_steps_taken + 1
        return self.weighted_sum_of_loss_tensors

    def translate(self):
        # print("translate velocity shape", type(velocity.numpy()), velocity.numpy()[:])
        self.atom_index = 0
        vectors = np.array([self.translate_atom(xyz) for xyz in self.velocity])

    def translate_atom(self, vector):
        # print("translate_atom", vector, type(vector), vector.shape)
        movement_vector = list(vector) # list(movement_vector)
        atom_selection_string = "id " + str(self.atom_index)
        cmd.translate(movement_vector, atom_selection_string)
        self.atom_index = self.atom_index + 1

    def plot(self):
        self.path_to_loss_png = "./results/{}/{}-loss.png".format(self.params.run_time_stamp, self.params.run_time_stamp)
        mpl.style.use('seaborn')
        if not os.path.exists(self.path_to_loss_png):
            self.fig = plt.figure(figsize=(18,10))
            self.ax1 = self.fig.add_subplot(121)
            plt.suptitle("Bit Pharma Neuromax: " + self.params.run_time_stamp)
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            self.ax2 = self.fig.add_subplot(122)
            plt.title('change in loss per step')
            plt.ylabel('count')
            plt.xlabel('difference in loss')
        self.ax1.plot(self.list,c='b')
        # histogram
        difference_loss=[0]
        for i in range(1,len((self.history.history['loss']))):
           difference_loss.append(self.history.history['loss'][i]-self.history.history['loss'][i-1])
        positive_values = list(filter(lambda x: x >0, difference_loss))
        negative_values = list(filter(lambda x: x <=0, difference_loss))
        self.ax2.hist(positive_values ,color='r')
        self.ax2.hist(negative_values,color='g')
        plt.tight_layout()
        plt.savefig('./results/{}/{}-loss.png'.format(self.params.run_time_stamp, self.params.run_time_stamp))

    def save(self):
        self.saver.save(file_prefix=self.checkpoint_prefix)

    def restore(self):
        self.saver.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

# main
searcher = Searcher()
params = searcher.search()
input_loader = Input_Loader(params)
trainer = Trainer(params, searcher, input_loader)
trainer.train()

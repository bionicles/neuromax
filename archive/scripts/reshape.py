import os
from PIL import Image
from tqdm import tqdm
import concurrent.futures
import glob

data_train_path = "data/train"
labels = [f for f in os.listdir("data/train")]


def reshape(file_name):
    img=Image.open(file_name)
    img=img.resize((350, 150))
    img.save(file_name)


for label in labels:
	print("Reshaping files of ", label)
	files = glob.glob(data_train_path + "/" + label + "/" + "*.jpeg")
	with concurrent.futures.ProcessPoolExecutor(max_workers = 15) as executor: # use 15 cores
		executor.map(reshape, files)

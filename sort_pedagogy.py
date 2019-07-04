import pandas as pd
from pymol import cmd
import os
from tqdm import tqdm
pdb_path = "./pdbs"

data = pd.read_csv("./csvs/less-than-164kd-9-chains.csv")

num_atoms = []
print("Loading number of atoms.")
for pdb_file_name in tqdm(data.columns):
    cmd.delete("all")
    if len(pdb_file_name)!=4:
        pdb_file_name = pdb_file_name[1:]
    pdb_file_name = pdb_file_name.lower()
    pdb_path = "./pdbs/" + pdb_file_name + ".pdb"
    if not os.path.exists(pdb_path):
        print('fetching', pdb_file_name)
        cmd.fetch(pdb_file_name, path=os.path.join('.', 'pdbs'), type='pdb')
    cmd.load(pdb_path)
    num_atoms.append(cmd.count_atoms("all"))

data = data.transpose()
data['num_atoms'] = num_atoms
data = data.nsmallest(data.shape[0], 'num_atoms') # use nlargest for to get the inverse
data.drop("num_atoms", axis=1, inplace=True)
data = data.transpose()
data.to_csv('./csvs/small~>big less then 9 chains.csv', index=False)

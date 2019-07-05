import pybel
import os
#https://figshare.com/articles/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873
pdbs_path = "./reactions"
reactions = open("rsmi/2001Sep2016USPTOapplicationssmiles.rsmi")
if not os.path.exists(pdbs_path):
    os.makedirs(pdbs_path)
def smiles_to_pdb(reaction, path):
    formula_type_index = 0
    for formula in reaction:
        if formula_type_index == 0:
            formula_type = "_reagent"
        if formula_type_index == 1:
            formula_type = "_solvent"
        if formula == '':
            continue
        if formula_type_index == 2:
            formula_type = "_product"
        mymol = pybel.readstring("smi", formula)
        mymol.make3D()
        mymol.write("pdb", os.path.join(path, formula_type)+'.pdb')
        formula_type_index += 1

counter = 0
product, solvent, reagent = [], [], []
for reaction in reactions:
    reaction = reaction.split(">")
    reaction[-1] = reaction[-1].partition(" ")
    reaction[-1] = reaction[-1][0]
    reaction_path = os.path.join(pdbs_path, 'reaction_'+str(counter))
    if not os.path.exists(reaction_path):
        os.makedirs(reaction_path)
    try:
        smiles_to_pdb(reaction, reaction_path)
        print("Converting reaction ", counter, " to pdb")
    except:
        print("failed to convert smiles to pdb")
    counter += 1

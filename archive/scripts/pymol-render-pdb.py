# pymol-render-pdb
# bion howard - 9 june 2018
# why?: render a bunch of protein screenshots

# can install schrodinger pymol with conda (warning: incentive version)
# pymol pre-compiled available at https://www.lfd.uci.edu/~gohlke/pythonlibs/#pymol
# pymol -qc ./pymol-render-pdb.py ("pymol -qc" runs without gui if desired)

from time import sleep
import pymol, os, csv
import random

pymol.finish_launching()
from pymol import cmd

def pickAColor():
    colors = ["green","red","yellow","blue","orange","marine","violet"]
    color = random.choice(colors)
    return color

def screenshotProteins():
    # proteinsPath = ".\pdbs"

    # set style parameters
    # cmd.set("ray_opaque_background", 0)
    # cmd.remove("solvent")
    cmd.set("ambient", 0.3)
    cmd.set("antialias", 1)
    cmd.bg_color("white")
    # cmd.set("direct", 1.0)
    # cmd.set("ribbon_radius", 0.2)
    # cmd.set("cartoon_highlight_color", "grey50")
    # cmd.set("ray_trace_mode", 1)
    # cmd.set("stick_radius", 0.2)
    # cmd.set("mesh_radius", 0.02)

    # loop thru folders
    # for dir in os.walk(proteinsPath):
        # proteinPath = dir[0]

        # loop thru pdbs
        # for file in os.listdir(proteinPath):
            # if file.endswith('.pdb'):

    csvFileName = "docked-protein-homomers.csv"

    # to loop through csv

    file = open(csvFileName, "r")
    reader = csv.reader(file, delimiter=",")
    k = 1
    for row in reader:
        for item in row:
            print(k, item)

            cmd.reinitialize()

            # load the pdb file
            cmd.fetch(item, path="./pdbs",type='pdb')
            #pdbPath= "./pdbs/" + item + ".cif"
            #cmd.load(pdbPath)
            color = pickAColor()
            cmd.color(color)
            cmd.show("cartoon")
            cmd.remove("solvent")

            # take a screenshot
            screenshotFileName = item + ".png"
            screenshotPath = os.path.join('screenshots', screenshotFileName)
            cmd.png(screenshotPath, 128, 128,ray=1)

            # clear
            cmd.delete("all")


            k = k+1

# to run the function
cmd.extend("screenshotProteins", screenshotProteins())

import prody as prd
import pandas as pd
import numpy as np
import easygui as eg
import multiprocessing.dummy as mp

#Select and parse a PDB
file = eg.fileopenbox()
structure = prd.parsePDB(file)

#Generates a list of all unique residues that will be calculated for
init_core_res = []
for resnum in structure.getResnums():
    if init_core_res.__contains__(resnum):
        continue
    elif resnum != 1 and resnum != max(structure.getResnums()):
        init_core_res.append(resnum)

#Loops the above residue list to create one "residue list" per frame (coordinate set)
frame_list = np.linspace(0,len(structure.getCoordsets())-1,num=len(structure.getCoordsets()))
core_res = []
for frame in range(len(frame_list)):
    core_res.extend(init_core_res)

#Returns the structure for a given frame
def get_str(index):
    structure.setACSIndex(index)
    return structure

#Pairs phi/psi lists by residue for a given frame
def phi_psi_pair(phis, psis):
    
    return 1

#Generates lists of phi and psi angles for all frames
with mp.Pool() as pool:
    frame_list = pool.map(lambda x: int(x), frame_list)
    structure_list = pool.map(lambda x: get_str(x), frame_list)
    hier_list = pool.map(lambda x: prd.HierView(x), structure_list)
    res_list = map(lambda x, y: x.getResidue('A', y), hier_list, core_res)
    phi_list = pool.map(lambda x: prd.calcPhi(x), res_list)
    res_list = map(lambda x, y: x.getResidue('A', y), hier_list, core_res)
    psi_list = pool.map(lambda x: prd.calcPsi(x), res_list)
    phi_list = list(phi_list)
    psi_list = list(psi_list)

#Generates the columns and rows for a dataframe to store all angles
clmns = []
rows = {}
for i in range(len(init_core_res)):
    clmns.append('phi' f'{i+1}')
    clmns.append('psi' f'{i+1}')

#Generates a dataframe and stores all angle values
angles = pd.DataFrame.from_dict(rows, orient = 'index', columns = clmns)
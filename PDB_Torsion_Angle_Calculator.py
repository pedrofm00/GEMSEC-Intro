import Bio.PDB as bpdb
import prody as prd
import numpy as np
import pandas as pd
import easygui as eg
import multiprocessing.dummy as mp

work_dir = eg.diropenbox() + '\\'
name = input('PDB to be Analyzed (do not include .pdb extension): ')
structure = prd.parsePDB(work_dir + name + '.pdb')
back_only = prd.writePDB(work_dir + name + "_backbone.pdb", structure.select('name N CA C'))

#Parse through the backbone
parser = bpdb.PDBParser()
backbone = parser.get_structure(name, back_only)

frame = 1
clmns = []
rows={}
for i in range(10):
    clmns.append('phi' f'{i+1}')
    clmns.append('psi' f'{i+1}')

model_list = bpdb.Selection.unfold_entities(backbone, 'M')
with mp.Pool() as pool:    
    chain_list = pool.map(lambda x: x['A'], model_list)
    poly_list = pool.map(lambda x: bpdb.Polypeptide.Polypeptide(x), chain_list)
    angle_list = pool.map(lambda x: x.get_phi_psi_list(), poly_list)
    rowstuff = pool.map(lambda x: np.reshape(x,[1,len(x)*2])[0][2:-2] * (180/np.pi), angle_list)
    rowlist = list(rowstuff)

#Generate a dataframe, store angles, and save to a csv in a chosen directory
angles_by_frame = pd.DataFrame(columns = np.linspace(1,22,num = 22))
angles_by_frame = pd.DataFrame(rowlist,index=np.linspace(1,len(rowlist),num=len(rowlist)),columns=clmns)
angles_by_frame.to_csv(work_dir + name + '.csv')
import Bio.PDB as bpdb
import numpy as np
import pandas as pd

parser = bpdb.PDBParser()
work_dir = 'E:/MD/GrBP5_YRRY/5ns_Simulations/pH9/310K/'
structure = parser.get_structure('YRRY_ph9_310K', work_dir + 'GrBP5_YRRY_pH9_310K_NVT_5ns_BackboneOnly.pdb')

angles_by_frame = pd.DataFrame(columns = np.linspace(1,22,num = 22))

frame = 1
rows = {}
clmns = []
for i in range(11):
    clmns.append('psi' f'{i+1}')
    clmns.append('phi' f'{i+1}')

for model in structure.get_models():
    for chain in model:
        poly = bpdb.Polypeptide.Polypeptide(chain)
        angles = poly.get_phi_psi_list()
        rows['Frame ' f'{frame}'] = np.reshape(angles, [1,len(angles)*2])[0][1:-1] * (180/np.pi)
        angles_by_frame = pd.DataFrame.from_dict(rows, orient = 'index', columns = clmns)
    frame += 1

print(angles_by_frame)
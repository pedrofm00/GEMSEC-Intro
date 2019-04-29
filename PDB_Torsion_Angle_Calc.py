#CURRENTLY CALCULATES ALL TORSION ANGLES FOR A SINGLE FRAME CONTAINING ONLY BACKBONE ATOMS
#FURTHER IMPLEMENTATIONS:
    #Allow for all frames from a simulation to be analyzed individually and organized
    #Allow for all pdbs to be provided, not just backbone only types

import Bio.PDB as bpdb
import pandas as pd

#Creating a Structure from a PDB (backbone only PDB needed)
parser = bpdb.PDBParser()
work_dir = 'E:/MD/GrBP5_YRRY/5ns_Simulations/pH9/310K/'
structure = parser.get_structure('YRRY_ph9_310K', work_dir + 'GrBP5_YRRY_pH9_310K_NVT_5ns_BackboneOnly_frame497.pdb')

# =============================================================================
# #Display all atoms in the chain
# atom_count = 0
# for atom in structure.get_atoms():
#     atom_count += 1
#     print(atom)
# print('Number of atoms present:')
# print(atom_count)
# =============================================================================

#Create a DataFrame to store all angles
calc_count = [1,2,3,4,5,6] #Count of calculated angles (of each kind) for ONE frame backbone
types = ['psi', 'phi']
angle_df = pd.DataFrame(index = calc_count, columns = types)

#Generate List of all atoms in backbone frame
atom_list = bpdb.Selection.unfold_entities(structure, 'A')
#print(atom_list)

#Calculate the Dihedral Angles
def Angle_Calc(atom1, atom2, atom3, atom4):
    v1 = atom1.get_vector()
    v2 = atom2.get_vector()
    v3 = atom3.get_vector()
    v4 = atom4.get_vector()
    return bpdb.calc_dihedral(v1, v2, v3, v4)

#Loop through all sequences for phi/psi angles
index = 0
for i in range(angle_df.shape[0]):
    if index < (len(atom_list)):
        a1 = atom_list[index]
        a2 = atom_list[index + 1]
        a3 = atom_list[index + 2]
        a4 = atom_list[index + 3]
        a5 = atom_list[index + 4]
        a6 = atom_list[index + 5]
        angle_df.iloc[i, 0] = Angle_Calc(a1, a2, a3, a4)
            #a1 is N, a2 is CA, a3 is C, a4 is N
        angle_df.iloc[i, 1] = Angle_Calc(a3, a4, a5, a6)
            #a3 is C, a4 is N, a5 is CA, a6 is C
        print(a1, a2, a3, a4, a5, a6)
    index += 6

print(angle_df)
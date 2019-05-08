#End-to-end MD Trajectory Analysis
#Takes in a pdb, then returns PCA and Ramachandran plots for this trajectory

import easygui as eg

import PDB_Torsion_Angle_Calculator as tor_calc
import MD_PCA_Clustering as md_pca
import Ramachandran_Plotting as ram_plt

#Select Working Directory and File to be Analyzed
work_dir = eg.diropenbox() + '\\'
file = input('Name of PDB to be Analyzed (do not include .pdb extension): ')

#Calculate Angles and store in dataframe
back = tor_calc.get_backbone(work_dir, file)
angles = tor_calc.get_angles(back)

#Complete PCA on angles
pca_data = md_pca.scale_data(angles)
md_pca.pca(pca_data)

#Create Ramachandran Plots for Residues
one_or_all = input('One Residue Ramachandran or All (enter "one" or "all"): ').lower()
if one_or_all == 'one':
    res = input('Which Residue (1-10): ')
    ram_plt.plt_one(angles, res)
elif one_or_all == 'all':
    ram_plt.plt_all(angles)
else:
    raise Exception('Must select either "one" or "all" residues to plot.')
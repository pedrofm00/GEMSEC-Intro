#End-to-end MD Trajectory Analysis

import easygui as eg
import PDB_Torsion_Angle_Calculator as tor_calc
import MD_PCA_Clustering as md_pca
import Ramachandran_Plotting as ram_plt
import Gaussian_Mixture_Clustering as gmc
import GM_Cluster_Transition_Counter as clst_cnt

#Select Working Directory and File to be Analyzed
work_dir = eg.diropenbox() + '\\'
file = input('Name of PDB to be Analyzed (do not include .pdb extension): ')

#Calculate Angles and store in dataframe
back = tor_calc.get_backbone(work_dir, file)
angles = tor_calc.get_angles(back)

#Complete PCA on angles
pca_data, pca_obj = md_pca.pca(angles, file, work_dir)

#Find and Display Loading Scores for PCs
pcnum = input('PC to display Loading Scores for: ')
score_cnt = input('Number of Scores to Display: ')
loading_scores = md_pca.load_score(pca_obj, int(pcnum), int(score_cnt))

#Gaussian Clustering
clust = input('Number of Clusters: ')
clust = int(clust)
dimen = input('Number of Dimensions: ')
dimen = int(dimen)
predictions, means, covar = gmc.cluster_PCA(file, pca_data, clust, dimen, work_dir)

#Calculate Transitions Between Clusters
transitions = clst_cnt.count_unique_trans(predictions)
total_transitions = clst_cnt.count_trans(predictions)
trans_freq = clst_cnt.transition_frequency(total_transitions, transitions)
clst_cnt.plot_tf(transitions, trans_freq, work_dir, file)

#Create Ramachandran Plots for Residues
one_or_all = input('One Residue Ramachandran or All (enter "one" or "all"): ').lower()
if one_or_all == 'one':
    res = input('Which Residue (1-10): ')
    ram_plt.plt_one(angles, res, work_dir, file)
elif one_or_all == 'all':
    ram_plt.plt_all(angles, work_dir, file)
else:
    raise Exception('Must select either "one" or "all" residues to plot.')
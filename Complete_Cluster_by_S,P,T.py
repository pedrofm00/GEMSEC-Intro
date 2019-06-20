#Sequence, pH, temp Based Clustering

#imports
import easygui as eg
import pandas as pd
import numpy as np
import Multi_PDB_Clustering as mpdbc
import Gaussian_Mixture_Clustering as gmc
import Ramachandran_Plotting as rmp
import matplotlib.pyplot as plt

sequences = ['FSSF', 'YDDY', 'YRRY']
pHs = ['pH3', 'pH7', 'pH9', 'pH11']
temperatures = ['280K', '290K', '300K', '310K', '320K']

#Select directory containing all other files
work_dir = eg.diropenbox() + '\\'

#Select Directory to save files to
save_dir = eg.diropenbox() + '\\'

names = []
locs = []

for pH in pHs:
    for sequence in sequences:
        for temp in temperatures:
            file_name = 'GrBP5_' + sequence + '_' + pH + '_' + temp + '_NVT_5ns'
            file_loc = (work_dir + 'GrBP5_' + sequence + '\\' + '5ns_Simulations'
                        + '\\' + pH + '\\' + temp + '\\')
            names.append(file_name)
            locs.append(file_loc)

comb_angles = mpdbc.get_combo_angles(names, locs, len(names))
cluster_pca = mpdbc.md_pca.pca(comb_angles, 'Total Clustered PCA', save_dir, '3d')
gmm = mpdbc.GMM_process(cluster_pca[0], save_dir)
cf_df = gmc.clust_prop(gmm[1], gmm[0][0], names)
cf_df.to_csv(path_or_buf = save_dir + '\\Simulation Frequencies per Cluster.csv')
mpdbc.transition_plot(gmm[0], save_dir)
plt.close()
mpdbc.plt_by_clust(gmm[1], gmm[0][0], names, cluster_pca[0], save_dir,
                   'PCA Clusters by Simulation')
plt.close()
rmp.plt_clust(gmm[0], comb_angles, save_dir)
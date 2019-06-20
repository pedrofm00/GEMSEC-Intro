#Manual PDB Clustering

import pandas as pd
import easygui as eg
import Multi_PDB_Clustering as mpdbc
import Gaussian_Mixture_Clustering as gmc
import Ramachandran_Plotting as rmp

#Select files to be analyzed
n = int(input('How many pdbs would you like to cluster?\n'))
files = []
for i in range(1, n+1):
    files.append(eg.fileopenbox())
fn, wd = mpdbc.get_names_dirs(files)

#Select folder to save files to
save_dir = eg.diropenbox() + '\\'

#Analyze
comb_angles = mpdbc.get_combo_angles(fn, wd, n)
cluster_pca = mpdbc.md_pca.pca(comb_angles, 'PCA of Peptide Torsion Angles',
                               save_dir, '2d')
gmm = mpdbc.GMM_process(cluster_pca[0], save_dir)
gmc.clust_prop(gmm[1], gmm[0][0], files).to_csv(path_or_buf = save_dir +
              '\\Simulation Densities per Cluster.csv')
mpdbc.plt_by_sim(gmm[1], comb_angles, cluster_pca[0], save_dir,
                   'PCA Clusters by Simulation')
mpdbc.transition_plot(gmm, save_dir)
rmp.plt_clust(gmm[0], comb_angles, save_dir)
load_scores = mpdbc.get_load_scores(cluster_pca[1], save_dir)
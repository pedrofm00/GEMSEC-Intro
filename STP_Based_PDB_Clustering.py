#Temperature Based Clustering

#imports
import easygui as eg
import pandas as pd
import numpy as np
import Multi_PDB_Clustering as mpdbc
import matplotlib.pyplot as plt

sequences = ['FSSF', 'YDDY', 'YRRY']
pHs = ['pH3', 'pH7', 'pH9', 'pH11']
temperatures = ['280K', '290K', '300K', '310K', '320K']

#Select directory containing all other files
work_dir = eg.diropenbox() + '\\'

#Select Directory to save files to
save_dir = eg.diropenbox() + '\\'

for sequence in sequences:
    for temp in temperatures:
        names = []
        locs = []
        for pH in pHs:
            file_name = 'GrBP5_' + sequence + '_' + pH + '_' + temp + '_NVT_5ns'
            file_loc = (work_dir + 'GrBP5_' + sequence + '\\' + '5ns_Simulations'
                        + '\\' + pH + '\\' + temp + '\\')
            names.append(file_name)
            locs.append(file_loc)

        comb_angles = mpdbc.get_combo_angles(names, locs, len(names))
        cluster_pca = mpdbc.md_pca.pca(comb_angles, 'Clustered PCA - ' +
                                       sequence + '_' + temp, save_dir)
        gmm = mpdbc.GMM_process(cluster_pca[0], save_dir, seq = sequence, temp = temp)
        mpdbc.transition_plot(gmm, save_dir, seq = sequence, temp = temp)
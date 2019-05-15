#Clustering and transition counts for multiple PDB files

import pandas as pd
import easygui as eg
import multiprocessing.dummy as mp
import PDB_Torsion_Angle_Calculator as tor_calc
import MD_PCA_Clustering as md_pca
import Gaussian_Mixture_Clustering as gmc
import GM_Cluster_Transition_Counter as clst_cnt

#Select files to be analyzed
n = int(input('How many pdbs would you like to cluster?\n'))
files = []
for i in range(1, n+1):
    print("Select File " + f'{i}' + ': ')
    files.append(eg.fileopenbox())

#Parse file and directory names out of each file selection
file_names = []
work_dirs = []
for i in range(len(files)):
    j = files[i].rindex('\\') + 1
    file_names.append(files[i][j:-4])
    work_dirs.append(files[i][:j])
    
#Select folder to save files to
print("Select the Save Directory: ")
save_dir = eg.diropenbox() + '\\'

#Get all datapoints needed for PCA
with mp.Pool() as pool:
    backbone_list = pool.map(lambda x: tor_calc.get_backbone(work_dirs[x], file_names[x]),
                             range(n))
    angle_list = pool.map(lambda x: tor_calc.get_angles(x), backbone_list)
#    combined_angles = pd.concat(angle_list)
    pca_list = map(lambda x, y: md_pca.pca(x, file_names[y], save_dir)[0], angle_list, range(n))
    pca_list = list(pca_list)
    combined_pcas = pd.concat(pca_list)

#Do PCA and GMM on data
#cluster_pca = md_pca.pca(combined_angles, 'Clustered PCA', save_dir)

clust_count = int(input('How many GMM clusters?\n'))
dim = int(input('How many GMM dimensions?\n'))
multi_gmm = gmc.cluster_PCA('Multi-PDB GMM', combined_pcas, clust_count, dim, save_dir)

#Calculate the transitions between clusters
unique_shifts = clst_cnt.count_unique_trans(multi_gmm[0])
total_shifts = clst_cnt.count_trans(multi_gmm[0])
tf = clst_cnt.transition_frequency(total_shifts, unique_shifts)
clst_cnt.plot_tf(unique_shifts, tf, save_dir, 'Multi_PDB Cluster Transitions')
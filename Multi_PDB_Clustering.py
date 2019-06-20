#Clustering and transition counts for multiple PDB files
import pandas as pd
import multiprocessing.dummy as mp
import matplotlib.pyplot as plt
import PDB_Torsion_Angle_Calculator as tor_calc
import MD_PCA_Clustering as md_pca
import Gaussian_Mixture_Clustering as gmc
import GM_Cluster_Transition_Counter as clst_cnt

#Parse file and directory names out of each file selection
def get_names_dirs(files):
    file_names = []
    work_dirs = []
    for i in range(len(files)):
        j = files[i].rindex('\\') + 1
        file_names.append(files[i][j:-4])
        work_dirs.append(files[i][:j])
    return file_names, work_dirs

#Create a list of all backbone angles across all conditions
def get_combo_angles(file_names, work_dirs, n):
    with mp.Pool() as pool:
        backbone_list = pool.map(lambda x: tor_calc.get_backbone(work_dirs[x], file_names[x]),
                                 range(n))
        angle_list = pool.map(lambda x: tor_calc.get_angles(x), backbone_list)
        combined_angles = pd.concat(angle_list)
    return combined_angles

#Complete GMM
def GMM_process(pca_df, save_dir, seq = '', pH = '', temp = ''):
#    optimal_clusters = gmc.opt_clust_dim_counts(pca_df)    
#    if seq != '' and pH != '' and temp != '':
#        multi_gmm = gmc.cluster_PCA('Multi-PDB GMM - ' + seq + '_' + pH + '_' + temp,
#                                    pca_df, optimal_clusters, 3, save_dir)
#    elif pH == '':
#            multi_gmm = gmc.cluster_PCA('Multi-PDB GMM - ' + seq + '_' + temp,
#                                pca_df, optimal_clusters, 3, save_dir)
#    elif temp == '':
#        multi_gmm = gmc.cluster_PCA('Multi-PDB GMM - ' + seq + '_' + pH,
#                                    pca_df, optimal_clusters, 3, save_dir)
#    elif seq == '':
#            multi_gmm = gmc.cluster_PCA('Multi-PDB GMM - ' + pH + '_' + temp,
#                                pca_df, optimal_clusters, 3, save_dir)
#    else:
#        multi_gmm = gmc.cluster_PCA('Multi-PDB GMM', pca_df, optimal_clusters, 3, save_dir)
    
    clust = int(input('Number of Clusters: '))
    dims = int(input('Number of Dimensions: '))
    
    if seq != '' and pH != '' and temp != '':
        multi_gmm = gmc.cluster_PCA('Multi-PDB GMM - ' + seq + '_' + pH + '_' + temp,
                                    pca_df, clust, dims, save_dir)
    elif seq != '' and pH == '' and temp != '':
        multi_gmm = gmc.cluster_PCA('Multi-PDB GMM - ' + seq + '_' + temp,
                                    pca_df, clust, dims, save_dir)
    elif seq != '' and pH != '' and temp == '':
        multi_gmm = gmc.cluster_PCA('Multi-PDB GMM - ' + seq + '_' + pH,
                                    pca_df, clust, dims, save_dir)
    elif seq == '' and pH != '' and temp != '':
        multi_gmm = gmc.cluster_PCA('Multi-PDB GMM - ' + pH + '_' + temp,
                                    pca_df, clust, dims, save_dir)
    elif seq != '' and pH == '' and temp == '':
        multi_gmm = gmc.cluster_PCA('Multi-PDB GMM - ' + seq, pca_df, clust, dims, save_dir)
    elif seq == '' and pH == '' and temp != '':
        multi_gmm = gmc.cluster_PCA('Multi-PDB GMM - ' + temp, pca_df, clust, dims, save_dir)
    elif seq == '' and pH != '' and temp == '':
        multi_gmm = gmc.cluster_PCA('Multi-PDB GMM - ' + pH, pca_df, clust, dims, save_dir)        
    else:
        multi_gmm = gmc.cluster_PCA('Multi-PDB GMM', pca_df, clust, dims, save_dir)
    return multi_gmm, clust

#Calculate the transitions between clusters of the GMM
def transition_plot(multi_gmm, save_dir, seq = '', pH = '', temp = ''):
    unique_shifts = clst_cnt.count_unique_trans(multi_gmm[0])
    total_shifts = clst_cnt.count_trans(multi_gmm[0])
    tf = clst_cnt.transition_frequency(total_shifts, unique_shifts)
    
    if seq != '' and pH != '' and temp != '':
        clst_cnt.plot_tf(unique_shifts, tf, save_dir, seq + '_' + pH + '_' + temp)
    elif seq != '' and pH == '' and temp != '':
        clst_cnt.plot_tf(unique_shifts, tf, save_dir, seq + '_' + temp)
    elif seq != '' and pH != '' and temp == '':
        clst_cnt.plot_tf(unique_shifts, tf, save_dir, seq + '_' + pH)
    elif seq == '' and pH != '' and temp != '':
        clst_cnt.plot_tf(unique_shifts, tf, save_dir, pH + '_' + temp)
    elif seq != '' and pH == '' and temp == '':
        clst_cnt.plot_tf(unique_shifts, tf, save_dir, seq)
    elif seq == '' and pH == '' and temp != '':
        clst_cnt.plot_tf(unique_shifts, tf, save_dir, temp)
    elif seq == '' and pH != '' and temp == '':
        clst_cnt.plot_tf(unique_shifts, tf, save_dir, pH)        
    else:
        clst_cnt.plot_tf(unique_shifts, tf, save_dir, 'Multi-PDB')

#Collect Loading Scores
def get_load_scores(PCA, save_dir):
    PC = input('Which PC to Gather Loading Scores for (or "all"):\n')
    top = int(input('How many scores to gather (top x):\n'))
    loading = list(md_pca.load_score(PCA, PC, top))
    loading_df = pd.DataFrame(loading)
    loading_df.to_csv(save_dir + 'Loading Scores for Component - ' + PC + '.csv')
    return loading

def plt_by_sim(nc, ang_df, pca_df, save_dir, fname):
    colors = ['black', 'gold', 'teal', 'violet', 'lightcoral', 'red', 'salmon',
              'sienna', 'darkorange','tan', 'goldenrod', 'olive', 'greenyellow',
              'darkseagreen','limegreen', 'darkgreen', 'turquoise', 'deepskyblue',
              'dodgerblue', 'navy', 'darkviolet']
    
    divs = []
    for frame in range(len(ang_df)):
        if ang_df.index[frame] == 0.0:
            divs.append(frame)

    min_f = 0
    for i in range(len(divs)):
        max_f = divs[i] - 1
        plt.scatter(pca_df.PC1[min_f:max_f], pca_df.PC2[min_f:max_f], 
                    color = colors[i-1], s = 0.05)
        min_f = max_f + 2
    
    plt.title('PCA - Colored by Simulation')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(save_dir + fname + '.png', pad_inches = 0)
    plt.close()
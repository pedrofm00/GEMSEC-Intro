#Generate Ramachandran Plots for Peptide Residues

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import multiprocessing.dummy as mp

#Generates a plot for one residue
def plt_one(angles, res, work_dir, name):
    #Load Angles
    phis = angles['phi' + res]
    psis = angles['psi' + res]
    
    #Calculate point density
    phipsi = np.vstack([phis, psis])
    density = stat.gaussian_kde(phipsi)(phipsi)
    
    #Create Scatter Plot
    fig, ax = plt.subplots()
    ax.scatter(phis, psis, c = density, s = 100, edgecolor = '')
    
    #Create Title, labels, and axes
    plt.title('Residue ' + res + ' Ramachandran')
    ax.set_xlabel("phi")
    ax.set_ylabel("psi")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.axhline(0, color = 'black')
    ax.axvline(0, color = 'black')
    
    plt.savefig(work_dir + name + ' - Residue ' + res + ' Ramachandran')
    plt.close()

#Generates plots for all residues
def plt_all(angles, work_dir, file):    
    res_list = np.linspace(1, len(angles.columns)//2, num = len(angles.columns)//2)
    with mp.Pool() as pool:
        res_list = pool.map(lambda x: int(x), res_list)
        res_list = pool.map(lambda x: str(x), res_list)
    for res in res_list:
        plt_one(angles, res, work_dir, file)
        
def plt_clust(gmm, angles, work_dir):
    nc = int(input('Number of Clusters: '))
    angles['Cluster'] = gmm[0]
    for i in range(nc):
        cluster_phi_psi = angles[angles['Cluster'] == i]
        plt_all(cluster_phi_psi, work_dir, 'Cluster ' + f'{i}')
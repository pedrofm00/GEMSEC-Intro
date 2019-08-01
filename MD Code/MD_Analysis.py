# -*- coding: utf-8 -*-
# =============================================================================
# Created on Thu Jul 11 10:56:42 2019
# 
# @author: pedro
# =============================================================================

class Clustering:
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
        #Import necessary packages
        import pandas as pd
        import multiprocessing.dummy as mp
        from MD_Analysis import Angle_Calc
        
        with mp.Pool() as pool:
            backbone_list = pool.map(lambda x: Angle_Calc.get_backbone(work_dirs[x], file_names[x]),
                                     range(n))
            angle_list = pool.map(lambda x: Angle_Calc.get_angles(x), backbone_list)
            combined_angles = pd.concat(angle_list)
        return combined_angles

class Angle_Calc:
    #Get a backbone from a pdb
    def get_backbone(work_dir, name):
        #Import necessary packages
        import prody as prd
        import Bio.PDB as bpdb
        
        #Parse the pdb file for its structure and then backbone
        structure = prd.parsePDB(work_dir + name + '.pdb')
        back_only = prd.writePDB(work_dir + name + "_backbone.pdb", structure.select('name N CA C'))
        
        #Parse through the backbone
        parser = bpdb.PDBParser()
        backbone = parser.get_structure(name, back_only)
    
        return backbone
    
    #Get phi/psi angles from a backbone pdb
    def get_phi_psi(backbone):
        #Import necessary packages
        import Bio.PDB as bpdb
        import pandas as pd
        import numpy as np
        import multiprocessing.dummy as mp
        
        #Get phi/psi angles from biopython
        model_list = bpdb.Selection.unfold_entities(backbone, 'M')
        with mp.Pool() as pool:
            chain_list = pool.map(lambda x: x['A'], model_list)
            poly_list = pool.map(lambda x: bpdb.Polypeptide.Polypeptide(x), chain_list)
            angle_list = pool.map(lambda x: x.get_phi_psi_list(), poly_list)
            rowstuff = pool.map(lambda x: np.reshape(x,[1,len(x)*2])[0][2:-2] * (180/np.pi), angle_list)
            rowlist = list(rowstuff)
        
        #Generate a dataframe and store angles
        clmns = []
        end_marks = []
        for i in range(10):
            clmns.append('phi' f'{i+1}')
            clmns.append('psi' f'{i+1}')
            end_marks.append('EoS')
            end_marks.append('EoS')
    
        angles_by_frame = pd.DataFrame(columns = np.linspace(1,20,num = 20))
        angles_by_frame = pd.DataFrame(rowlist,index=np.linspace(1,len(rowlist),num=len(rowlist)),columns=clmns)
        end_marks = pd.DataFrame(end_marks, index = clmns)
        angles_by_frame = angles_by_frame.append(end_marks.T)

        return angles_by_frame
    
    #Define all possible chi angles for each residue
    #Credit:
        # Copyright (c) 2014 Lenna X. Peterson, all rights reserved
        # lenna@purdue.edu
    def __gen_chi_list():
        chi_atoms = dict(
            chi1=dict(
                ARG=['N', 'CA', 'CB', 'CG'],
                ASN=['N', 'CA', 'CB', 'CG'],
                ASP=['N', 'CA', 'CB', 'CG'],
                CYS=['N', 'CA', 'CB', 'SG'],
                GLN=['N', 'CA', 'CB', 'CG'],
                GLU=['N', 'CA', 'CB', 'CG'],
                HIS=['N', 'CA', 'CB', 'CG'],
                ILE=['N', 'CA', 'CB', 'CG1'],
                LEU=['N', 'CA', 'CB', 'CG'],
                LYS=['N', 'CA', 'CB', 'CG'],
                MET=['N', 'CA', 'CB', 'CG'],
                PHE=['N', 'CA', 'CB', 'CG'],
                PRO=['N', 'CA', 'CB', 'CG'],
                SER=['N', 'CA', 'CB', 'OG'],
                THR=['N', 'CA', 'CB', 'OG1'],
                TRP=['N', 'CA', 'CB', 'CG'],
                TYR=['N', 'CA', 'CB', 'CG'],
                VAL=['N', 'CA', 'CB', 'CG1'],
            ),
            chi2=dict(
                ARG=['CA', 'CB', 'CG', 'CD'],
                ASN=['CA', 'CB', 'CG', 'OD1'],
                ASP=['CA', 'CB', 'CG', 'OD1'],
                GLN=['CA', 'CB', 'CG', 'CD'],
                GLU=['CA', 'CB', 'CG', 'CD'],
                HIS=['CA', 'CB', 'CG', 'ND1'],
                ILE=['CA', 'CB', 'CG1', 'CD1'],
                LEU=['CA', 'CB', 'CG', 'CD1'],
                LYS=['CA', 'CB', 'CG', 'CD'],
                MET=['CA', 'CB', 'CG', 'SD'],
                PHE=['CA', 'CB', 'CG', 'CD1'],
                PRO=['CA', 'CB', 'CG', 'CD'],
                TRP=['CA', 'CB', 'CG', 'CD1'],
                TYR=['CA', 'CB', 'CG', 'CD1'],
            ),
            chi3=dict(
                ARG=['CB', 'CG', 'CD', 'NE'],
                GLN=['CB', 'CG', 'CD', 'OE1'],
                GLU=['CB', 'CG', 'CD', 'OE1'],
                LYS=['CB', 'CG', 'CD', 'CE'],
                MET=['CB', 'CG', 'SD', 'CE'],
            ),
            chi4=dict(
                ARG=['CG', 'CD', 'NE', 'CZ'],
                LYS=['CG', 'CD', 'CE', 'NZ'],
            ),
            chi5=dict(
                ARG=['CD', 'NE', 'CZ', 'NH1'],
            ),
        )
        
        return chi_atoms
    
    #Calculate the chi angle between a set of four atoms from a residue
    def __calc_chi(residue, group):
        #Import Necessary Packages
        from Bio import PDB
        import scipy.constants as const
        self = Angle_Calc
        
        #Define all possible chi angles for each residue
        chi_atoms = self.__gen_chi_list()

        #Convert Needed data to appropriate formats
        res_atoms = PDB.Selection.unfold_entities(residue, 'A')
        res_name = residue.get_resname()
        atom_names = []
        for atom in res_atoms:
            atom_names.append(atom.get_name())
        
        #Gather all four atoms to calculate the angle for
        atom1 = res_atoms[atom_names.index(chi_atoms[group][res_name][0])].get_vector()
        atom2 = res_atoms[atom_names.index(chi_atoms[group][res_name][1])].get_vector()
        atom3 = res_atoms[atom_names.index(chi_atoms[group][res_name][2])].get_vector()
        atom4 = res_atoms[atom_names.index(chi_atoms[group][res_name][3])].get_vector()
        #Return the dihedral angle between the atoms
        return PDB.calc_dihedral(atom1, atom2, atom3, atom4)*(180/const.pi)

    #Get chi angles for each residue from a pdb
    def get_chi(file):
        #Import Necessary Packages
        from Bio import PDB
        import pandas as pd
        import multiprocessing.dummy as mp
        self = Angle_Calc
        
        #Define all possible chi angles for each residue
        chi_atoms = self.__gen_chi_list()
            
        #Import the pdb structure file
        parser = PDB.PDBParser()
        pep = parser.get_structure(file[:-4], file)
        
        #Get a list of each residue
        model_list = PDB.Selection.unfold_entities(pep, 'M')
        with mp.Pool() as pool:
            res_list = pool.map(lambda x: PDB.Selection.unfold_entities(x, 'R'), model_list)
        
        chi_dict = {"Residue": [], "Chi 1": [], "Chi 2": [], "Chi 3": [],
                    "Chi 4": [], "Chi 5": []}
        
        #Break down the list into individual residues
        for frame in res_list:
            for res in frame:
                res_name = res.get_resname()
                chi_dict["Residue"].append(res_name + ' - Frame ' + f'{res_list.index(frame) + 1}')
                
                if res_name in chi_atoms["chi5"]:
                    chi_dict["Chi 1"].append(self.__calc_chi(res, "chi1"))
                    chi_dict["Chi 2"].append(self.__calc_chi(res, "chi2"))
                    chi_dict["Chi 3"].append(self.__calc_chi(res, "chi3"))
                    chi_dict["Chi 4"].append(self.__calc_chi(res, "chi4"))
                    chi_dict["Chi 5"].append(self.__calc_chi(res, "chi5"))
                elif res_name in chi_atoms["chi4"]:
                    chi_dict["Chi 1"].append(self.__calc_chi(res, "chi1"))
                    chi_dict["Chi 2"].append(self.__calc_chi(res, "chi2"))
                    chi_dict["Chi 3"].append(self.__calc_chi(res, "chi3"))
                    chi_dict["Chi 4"].append(self.__calc_chi(res, "chi4"))
                    chi_dict["Chi 5"].append(0.0)
                elif res_name in chi_atoms["chi3"]:
                    chi_dict["Chi 1"].append(self.__calc_chi(res, "chi1"))
                    chi_dict["Chi 2"].append(self.__calc_chi(res, "chi2"))
                    chi_dict["Chi 3"].append(self.__calc_chi(res, "chi3"))
                    chi_dict["Chi 4"].append(0.0)
                    chi_dict["Chi 5"].append(0.0)
                elif res_name in chi_atoms["chi2"]:
                    chi_dict["Chi 1"].append(self.__calc_chi(res, "chi1"))
                    chi_dict["Chi 2"].append(self.__calc_chi(res, "chi2"))
                    chi_dict["Chi 3"].append(0.0)
                    chi_dict["Chi 4"].append(0.0)
                    chi_dict["Chi 5"].append(0.0)
                elif res_name in chi_atoms["chi1"]:
                    chi_dict["Chi 1"].append(self.__calc_chi(res, "chi1"))
                    chi_dict["Chi 2"].append(0.0)
                    chi_dict["Chi 3"].append(0.0)
                    chi_dict["Chi 4"].append(0.0)
                    chi_dict["Chi 5"].append(0.0)
                else:
                    chi_dict["Chi 1"].append(0.0)
                    chi_dict["Chi 2"].append(0.0)
                    chi_dict["Chi 3"].append(0.0)
                    chi_dict["Chi 4"].append(0.0)
                    chi_dict["Chi 5"].append(0.0)
        
        chi_df = pd.DataFrame.from_dict(chi_dict)
        return chi_df
    
    def get_sin_cos(angle_df):
        #Import necessary packages
        import pandas as pd
        import numpy as np
        
        #Create the new dataframe
        sc_df = pd.DataFrame()
        
        #Apply sin and cos transformations to each column in angle_df
        for col in angle_df.columns:
            if col == 'Residue':
                sc_df.insert(0, 'Residue', angle_df['Residue'])
            else:
                sc_df['Sin - ' + col] = angle_df[col].map(lambda x: np.sin(x))
                sc_df['Cos - ' + col] = angle_df[col].map(lambda x: np.cos(x))
        
class PCA_Analysis:
    #Function to complete Principle Component Analysis on a given dataset
    def pca(angles, file_name, dir_name, graph_type):
        #Import necessary packages
        import pandas as pd
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn import preprocessing
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        #Get all the PCA components from the data
        pca = PCA()
        pc_angles = angles.drop(index = 0.0)
        data = preprocessing.scale(pc_angles)
        pca.fit(data)
        pca_data = pca.transform(data)
        per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1)
        labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
        
        #Create a Scree Plot of the components
        plt.close()
        plt.bar(x = range(1, len(per_var) + 1), height = per_var,
                tick_label = labels, rotation = 'vertical')
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal Component')
        plt.title('Scree Plot')
        plt.savefig(dir_name + 'Scree Plot - ' + file_name + '.png')
        plt.show()
        plt.close()
        
        pca_df = pd.DataFrame(pca_data, columns = labels)
    
        #Generate the PCA Graph based on PC1 and PC2
        if graph_type == '2d':
            plt.scatter(pca_df.PC1, pca_df.PC2, s = 0.01)
            plt.title('Torsion Angle PCA Graph')
            plt.xlabel('PC1 - {0}%'.format(per_var[0]))
            plt.ylabel('PC2 - {0}%'.format(per_var[1]))
            plt.savefig(dir_name + '2D PCA - ' + file_name + '.png')
            plt.close()
            
        #Generate the PCA Graph Based on PC1, PC2, and PC3
        elif graph_type == '3d':
            ax = plt.axes(projection = '3d')
            ax.scatter3D(pca_df.PC1, pca_df.PC2, pca_df.PC3, s = 0.01,
                         depthshade = True)
            ax.set_xlabel('PC1 - {0}%'.format(per_var[0]))
            ax.set_ylabel('PC2 - {0}%'.format(per_var[1]))
            ax.set_zlabel('PC3 - {0}%'.format(per_var[2]))
            plt.savefig(dir_name + '3D PCA - ' + file_name + '.png')
            plt.close()
        
        else:
            raise Exception('Graph Type must be either "2d" or "3d".')
        
        return pca_df, per_var

    #Function to gather loading scores after PCA is completed
    def load_score(pca, PC, n = 3, bottom = False):
        #import necessary packages
        import pandas as pd
        import multiprocessing.dummy as mp
        
        #Gather and return loading scores for all PCs
        #Optional: provide "n" for how many top/bottom scores to display
        if PC.lower() == "all":
            with mp.Pool() as pool:
                #Collect all scores
                all_scores = pool.map(lambda x: pd.Series(pca.components_[x]),
                                 range(len(pca.components_)))
                #Sort the scores in descending order
                all_sorted_scores = pool.map(lambda x: x.abs().sort_values(ascending = False), all_scores)
                #Gather the top "n" components and their scores
                all_top_n = pool.map(lambda x: x[0:n].index.values, all_sorted_scores)
                all_top_n_scores = pool.map(lambda x: x[0:n].values, all_sorted_scores)
                top_LS = pd.DataFrame.from_dict({"PC": all_top_n, "Score": all_top_n_scores})
                #Gather the bottom "n" components and their scores
                if bottom:
                    all_bot_n = pool.map(lambda x: x[-n:].index.values, all_sorted_scores)
                    all_bot_n_scores = pool.map(lambda x: x[-n:].values, all_sorted_scores)
                    bot_LS = pd.DataFrame.from_dict({"PC": all_bot_n, "Score": all_bot_n_scores})
                    return top_LS, bot_LS
            return top_LS
        
        #Gather and return loading scores for a given PC
        else:
            PC = int(PC)
            loading_scores = pd.Series(pca.components_[PC])
            # sort loading scores by magnitude
            sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
            # get names
            top_n = sorted_loading_scores[0:n].index.values
            top_n_scores = sorted_loading_scores[0:n].values
            top_LS = pd.DataFrame.from_dict({"PC": top_n, "Score": top_n_scores})
            if bottom:
                    bot_n = pool.map(lambda x: x[-n:].index.values, sorted_loading_scores)
                    bot_n_scores = pool.map(lambda x: x[-n:].values, sorted_loading_scores)
                    bot_LS = pd.DataFrame.from_dict({"PC": bot_n, "Score": bot_n_scores})
                    return top_LS, bot_LS
            return top_LS
        
    def PC_den_plt(pca_df, wd, seq='', temp='', pH=''):
        #Import necessary packages
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        file = '\\PCA Density Plot'
        if seq != '':
            file += ' - ' + seq
            if temp != '':
                file += '_' + temp
            if pH != '':
                file += '_' + pH
        elif temp != '':
            file += ' - ' + temp
            if pH != '':
                file += '_' + pH
        elif pH != '':
            file += ' - ' + pH
    
        pc1 = pca_df.PC1
        pc2 = pca_df.PC2
    
        #For a single PC plot
        f = plt.figure()
        sns.kdeplot(pc1, pc2, shade = True, n_levels = 75, cmap = "Purples")
        
        #For Multiple PC plots
        #    f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2)
        #    
        #    sns.kdeplot(pc1, pc2, shade = True, n_levels = 75, cmap = "Purples", ax = ax1)
        #    sns.kdeplot(pc1, pc2, shade = True, n_levels = 75, cmap = "Blues", ax = ax2)
        #    sns.kdeplot(pc1, pc2, shade = True, n_levels = 75, cmap = "GnBu", ax = ax3)
        #    sns.kdeplot(pc1, pc2, shade = True, n_levels = 75, cmap = "Oranges", ax = ax4)

        f.savefig(wd + file, pad_inches = 0)
        plt.close()
        
    def gen_2d_PCA_gif(pca, wd, seq='', temp='', pH=''):
        #Import necessary packages
        import numpy as np
        import matplotlib.pyplot as plt
        import multiprocessing.dummy as mp
        import imageio
        import os
        
        #Appropriately name the file
        file = 'Evolution of Conformation in PC Space'
        if seq != '':
            file += ' - ' + seq
            if temp != '':
                file += '_' + temp
            if pH != '':
                file += '_' + pH
        elif temp != '':
            file += ' - ' + temp
            if pH != '':
                file += '_' + pH
        elif pH != '':
            file += ' - ' + pH
        
        #Break down PCA input data to relevant pieces
        pca_df = pca[0]
        per_var = np.round(pca[1].explained_variance_ratio_ * 100, decimals = 1)

        #Select time frames to highlight in the gif
        frame_list = np.linspace(0, len(pca_df), num = len(pca_df)/10)[:-1]
        with mp.Pool() as pool:    
            frames = pool.map(lambda x: int(x), frame_list)
        names = []
        #Generate all (progressive) scatter plot images
        for frame in frames:
            plt.scatter(pca_df.PC1[:frame], pca_df.PC2[:frame], c = 'b', s = 0.01)
            #Integrate a plt_by_sim functionality here to clarify sudden jumps
            plt.scatter(pca_df.PC1[frame], pca_df.PC2[frame], c = 'r')
            plt.xlim(-5, 8)
            plt.ylim(-5, 8)
            plt.xlabel('PC1 - {0}%'.format(per_var[0]))
            plt.ylabel('PC2 - {0}%'.format(per_var[1]))
            plt.title(file + ' - Frame ' + f'{frame}')
            plt.savefig(wd + file + ' - Frame ' + f'{frame}' + '.png', pad_inches = 0)
            plt.close()
            names.append(file + ' - Frame ' + f'{frame}' + '.png')
    
        #Concatenate the images into a gif
        with imageio.get_writer(wd + file + '.gif', mode = 'I',
                                duration = 0.005) as writer:
            for filename in names:
                image = imageio.imread(wd + filename)
                writer.append_data(image)
    
        #Delete all individual images
        for frame in frames:
            os.remove(wd + file + ' - Frame ' + f'{frame}' + '.png')

class GMM_Clustering:
    #Plot all points in a color clustered PC graph
    def plot_clusters(X, Y_, means, covariances, index, title, dirname):
        #Import necessary packages
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from scipy import linalg
        import numpy as np
        import itertools
        
        #Colors of clusters
        color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange', 'crimson', 'g', 'darkviolet',
                              'darkgoldenrod', 'teal', 'purple', 'burlywood'])

        #NEED COMMENTS
        splot = plt.subplot(1, 1, 1 + index)
        for i, (mean, covar, color) in enumerate(zip(
                means, covariances, color_iter)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], s=.1, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.xticks(())
        plt.yticks(())
        plt.title(title)
        plt.savefig(dirname + title + '.png')
    
    #Cluster PC points based on Gaussian Mixture Similarities
    def cluster_PCs(name, pca_df, n, dim, dir_name):
        #Import necessary packages
        from sklearn import mixture
        import plot_clusters
        import matplotlib.pyplot as plt
        
        #Use Gaussian Mixture Clustering to cluster PC data
        gmm = mixture.GaussianMixture(n_components = n).fit(pca_df.iloc[:,0:dim].values)
        #Plot the Clusters
        plot_clusters(pca_df.iloc[:,0:dim].values, gmm.predict(pca_df.iloc[:,0:dim].values), 
                     gmm.means_, gmm.covariances_, 0, 'Gaussian Mixture - ' + name, dir_name)
        
        #Collect a dataframe of the cluster each point is placed in
        predictions = gmm.predict(pca_df.iloc[:,0:dim].values)
        
        #Generate labels for a bar plot
        cats = []
        for i in range(n):
            cats.append(f'{i}')
        
        #Count how many points are in each cluster
        counts = {}
        for i in range(len(predictions)):
            key = f'{predictions[i]}'
            if key in counts:
                counts[key] += 1
            else:
                counts[key] = 1
    
        #Plot a cluster density bar chart
        plt.close()
        plt.bar(cats, height = counts.values())
        plt.title('Cluster Distribution - ' + name)
        plt.xlabel('Clusters')
        plt.ylabel('Number of Frames') 
        plt.savefig(dir_name + 'Cluster Density - ' + name + '.png')
    
        return predictions, gmm.means_, gmm.covariances_
    
    #Calculate the proportions of each cluster made up by each simulation
    def clust_prop(nc, pred, sims):
        #Import necessary packages
        import pandas as pd
        
        #
        n_sims = len(sims)
        sim_size = len(pred)/n_sims
        cp_d = {}
        cp_d['Simulation'] = sims
        for cluster in range(nc):
            frames = []
            for frame in range(len(pred)):
                if pred[frame] == cluster:
                    frames.append(frame)
        
            min_f = 0
            max_f = sim_size
            sim_frames = {}
            for i in range(n_sims):
                sim_frames[i] = []
                for f in frames:
                    if f>= min_f and f < max_f:
                        sim_frames[i].append(f)
                    min_f += sim_size
                    max_f += sim_size
        
            cp_d[cluster] = []
            for i in range(n_sims):
                cp_d[cluster].append(len(sim_frames[i])/len(frames))
    
        cp_df = pd.DataFrame.from_dict(cp_d, orient = 'index')
        return cp_df
    
class GMM_Transitions:
    def count_trans(pred):
        trans_count = 0
    
        for i in range(len(pred[:])):
            if i == len(pred[:]) - 1:
                continue
            elif pred[i+1] != pred[i]:
                trans_count += 1
        return trans_count
    
    def transition_frequency(total, unique):
        tf = []
        for key in list(unique.keys()):
            tf.append(unique[key]/total)
        return tf

    def plot_tf(trans, tf, wd, name):
        #Import necessary packages
        import matplotlib.pyplot as plt
        
        plt.close()
        plt.bar(x = list(trans.keys()), height = tf)
        plt.xticks(rotation = 75)
        plt.title('Transition Frequencies Between Clusters')
        plt.xlabel('Transition')
        plt.ylabel('Frequency')
        plt.savefig(wd + 'Transition Frequencies Between Clusters - ' + name)
        plt.close()
    
class Ramachandran:
    #Create a Ramachandran Plot for one set of phi/psi angles
    def plt_one(angles, res, work_dir, name):
        #Import Necessary Packages
        import numpy as np
        import matplotlib.pyplot as plt
        import scipy.stats as stat
        
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
        
        #Save the figure and close it
        plt.savefig(work_dir + name + ' - Residue ' + res + ' Ramachandran')
        plt.close()
        
    #Generates plots for all residues of the chain
    def plt_all(angles, work_dir, file):    
        #Import Necessary Packages
        import numpy as np
        import multiprocessing.dummy as mp
        import plt_one
        
        #Gather all amino acid residues to plot
        res_list = np.linspace(1, len(angles.columns)//2, num = len(angles.columns)//2)
        with mp.Pool() as pool:
            res_list = pool.map(lambda x: int(x), res_list)
            res_list = pool.map(lambda x: str(x), res_list)
        
        #Plot each residue individually
        for res in res_list:
            plt_one(angles, res, work_dir, file)
    
    #Generates plots for each cluster in PC space
    def plt_clust(gmm, angles, work_dir):
        #Import Necessary Packages
        import plt_all
        
        #Collects the number of clusters
        nc = gmm[1]
        #Drops all EoS lines
        clst_angles = angles.drop(index = 0.0)
        #Appends a column with predictions to each frame
        clst_angles['Clusters'] = gmm[0][0]
        #Gather the frames for each cluster, then plot all for each grouping
        for i in range(nc):
            cluster_phi_psi = clst_angles[clst_angles['Cluster'] == i]
            plt_all(cluster_phi_psi, work_dir, 'Cluster ' + f'{i}')
            
class Hilbert_Transform:
    ## ROT rotates and flips a quadrant appropriately.
    #  Parameters:
    #    Input, integer N, the length of a side of the square.  
    #    N must be a power of 2.
    #    Input/output, integer X, Y, the coordinates of a point.
    #    Input, integer RX, RY, ???
    def rot(n, x, y, rx, ry):
        if (ry == 0):
            #Reflect.
            if (rx == 1):
                x = n - 1 - x
                y = n - 1 - y
            #Flip.
            t = x
            x = y
            y = t   
        return x, y
    
    ## XY2D converts a 2D Cartesian coordinate to a 1D Hilbert coordinate.
    #  Discussion:
    #    It is assumed that a square has been divided into an NxN array of cells,
    #    where N is a power of 2.
    #    Cell (0,0) is in the lower left corner, and (N-1,N-1) in the upper 
    #    right corner.
    #  Parameters:
    #    integer M, the index of the Hilbert curve.
    #    The number of cells is N=2^M.
    #    0 < M.
    #    Input, integer X, Y, the Cartesian coordinates of a cell.
    #    0 <= X, Y < N.
    #    Output, integer D, the Hilbert coordinate of the cell.
    #    0 <= D < N * N.
    def xy2d(self, x,y):
        
        m = 10    # index of hilbert curve
        n = 1024    # number of boxes (2^m)
    
        xcopy = x
        ycopy = y
    
        d = 0
        n = 2 ** m
    
        s = ( n // 2 )
    
        while ( 0 < s ):
            if ( 0 <  ( abs ( xcopy ) & s ) ):
                rx = 1
            else:
                rx = 0
            if ( 0 < ( abs ( ycopy ) & s ) ):
                ry = 1
            else:
                ry = 0
            d = d + s * s * ( ( 3 * rx ) ^ ry )
            xcopy, ycopy = self.rot(s, xcopy, ycopy, rx, ry )
            s = ( s // 2 )
        return d

    def hilb_collapse(self, data_2d):
        #Import necessary packages
        import numpy as np
        import pandas as pd
        
        #Transform and round data to integer values into pixel space
        #   - adding 180 because our lowest phi/psi value possible is -180 and we
        #       want lowest value to be zero.
        #   - dividing by 1023 because we are using order 10 (0-1023 is 1024)        
        transformed_data = data_2d.apply(lambda x: np.round((x+180)/(360/1023), decimals=0))
        rounded_data = transformed_data.apply(np.int64)
        
        #Combine phi psi values into one column
        combined_data = pd.DataFrame(index=rounded_data.index)
        for i in [0,2,4,6,8,10,12,14,16,18]:
            combined_data['AA'+str(i)]=rounded_data.iloc[:,i:i+2].values.tolist()
        
        #####INCOMPLETE#####
# =============================================================================
#         #Convert 2d into 1d
#         hilbert_data = np.zeros((19986, 10))
#         for i in range(19986):
#             for j in range(10):
#                 hilbert_data[i, j] = self.xy2d(combined_data.iloc[i,j][0],combined_data.iloc[i,j][1])
# 
#         #Add index and column titles to hilbert data
#         hilbert_data=pd.DataFrame(hilbert_data,index=combined_data.index,columns=combined_data.columns)
# 
#         #Save
#         hilbert_data.to_csv('hd_' + name + '.csv')
# =============================================================================    

#class RMSD:
#    def get_rmsd(n, angles):
# -*- coding: utf-8 -*-
# =============================================================================
# Created on Thu Jul 11 10:56:42 2019
# 
# @author: pedro
# =============================================================================

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
    def get_angles(backbone):
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
    
        angles_by_frame = pd.DataFrame(columns = np.linspace(1,22,num = 22))
        angles_by_frame = pd.DataFrame(rowlist,index=np.linspace(1,len(rowlist),num=len(rowlist)),columns=clmns)
        end_marks = pd.DataFrame(end_marks, index = clmns)
        angles_by_frame = angles_by_frame.append(end_marks.T)

        return angles_by_frame
    
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
    
class RMSD:
    def get_rmsd(n, angles):
        
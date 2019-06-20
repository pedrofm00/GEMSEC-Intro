#Gaussian Mixture
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
from scipy import linalg
import numpy as np
import pandas as pd

# colors of clusters
color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange', 'crimson', 'g', 'darkviolet',
                              'darkgoldenrod', 'teal', 'purple', 'burlywood'])

# gaussian mixture
def plot_results(X, Y_, means, covariances, index, title, dirname):
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

# plotting
def cluster_PCA(name, pca_df, n, dim, dir_name):
    gmm = mixture.GaussianMixture(n_components=n).fit(pca_df.iloc[:,0:dim].values)
    plot_results(pca_df.iloc[:,0:dim].values, gmm.predict(pca_df.iloc[:,0:dim].values), 
             gmm.means_, gmm.covariances_, 0, 'Gaussian Mixture - ' + name, dir_name)

##### PREDICTIONS #############################################################

    predictions = gmm.predict(pca_df.iloc[:,0:dim].values)
    
    cats = []
    for i in range(n):
        cats.append(f'{i}')
    
    counts = {}
    for i in range(len(predictions)):
        key = f'{predictions[i]}'
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = 1
    
    plt.close()
    plt.bar(cats, height = counts.values())
    plt.title('Cluster Distribution - ' + name)
    plt.xlabel('Clusters')
    plt.ylabel('Number of Frames') 
    plt.savefig(dir_name + 'Cluster Density - ' + name + '.png')
    
    return predictions, gmm.means_, gmm.covariances_

def clust_prop(nc, pred, sims):
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

def opt_clust_dim_counts(pca_df):
    clusters = 1
    dims = 2
    curr_cov = 0
    last_cov = 1
    while np.round(curr_cov, decimals = 0) != np.round(last_cov, decimals = 0):
        if curr_cov != 0:
            last_cov = curr_cov
        gmm = mixture.GaussianMixture(n_components = clusters).fit(pca_df.iloc[:, 0:dims].values)
        curr_cov = np.sum(gmm.covariances_)
        clusters += 1
    return clusters

#Clustering (see slack)
#Try K-means too maybe
#RMSD Clustering as a sanity check (only if package exists)
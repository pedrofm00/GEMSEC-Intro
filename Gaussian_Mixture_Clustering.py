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
                              'darkorange'])

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
    
    plt.close()
    plt.hist(predictions, align='mid')
    plt.title('Cluster Histogram - ' + name)
    plt.xlabel('Clusters')
    plt.ylabel('Number of Frames') 
    plt.savefig(dir_name + 'Cluster Density - ' + name + '.png')
    
    return predictions, gmm.means_, gmm.covariances_
#plt.xticks([])
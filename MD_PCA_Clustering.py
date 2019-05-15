import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Scale the data to be appropriate for PCA
def scale_data(angle_data):
    scaled = preprocessing.scale(angle_data)
    return scaled

#Perform PCA on the data
def pca(angles, name, dir_name):
    #Get all the PCA components from the data
    pca = PCA()
    data = scale_data(angles)
    pca.fit(data)
    pca_data = pca.transform(data)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1)
    labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
    
    #Create a Scree Plot of the components
    plt.bar(x = range(1, len(per_var) + 1), height = per_var, tick_label = labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.savefig(dir_name + 'Scree Plot - ' + name + '.png')
    plt.show()
    plt.close()
    
    pca_df = pd.DataFrame(pca_data, columns = labels)
    
    graph_type = input('2d or 3d Graph: ')
    graph_type = graph_type.lower()
    #Generate the PCA Graph based on PC1 and PC2
    if graph_type == '2d':
        plt.scatter(pca_df.PC1, pca_df.PC2)
        plt.title('Torsion Angle PCA Graph')
        plt.xlabel('PC1 - {0}%'.format(per_var[0]))
        plt.ylabel('PC2 - {0}%'.format(per_var[1]))
        plt.savefig(dir_name + '2D PCA - ' + name + '.png')
        plt.close()

    #Generate the PCA Graph Based on PC1, PC2, and PC3
    elif graph_type == '3d':
        ax = plt.axes(projection = '3d')
        ax.scatter3D(pca_df.PC1, pca_df.PC2, pca_df.PC3)
        plt.savefig(dir_name + '3D PCA - ' + name + '.png')
        plt.close()
        
    else:
        raise Exception('Graph Type must be either "2d" or "3d".')
        
    return pca_df, pca

def load_score(pca, PC, n):
    # loading scores
    loading_scores = pd.Series(pca.components_[PC])

    # sort loading scores by magnitude
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)

    # get names
    top_n = sorted_loading_scores[0:n].index.values
    return top_n

def combine_pcas(pca1, pca2):
    pca1.append(pca2)
    return pca1
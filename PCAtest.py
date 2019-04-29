import pandas as pd
import numpy as np
import random as rd #not needed if working with real data
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

#Producing random data to do the PCA on, actual data would be parsed out of another file
genes = ['gene' + str(i) for i in range(1,101)]

wt = ['wt' + str(i) for i in range(1,6)]
ko = ['ko' + str(i) for i in range(1,6)]

data = pd.DataFrame(columns=[*wt, *ko], index=genes)

for gene in data.index:
	data.loc[gene, 'wt1' : 'wt5'] = np.random.poisson(lam=rd.randrange(10,1000), size = 5)
	data.loc[gene, 'ko1' : 'ko5'] = np.random.poisson(lam=rd.randrange(10,1000), size = 5)

print(data.head())
print(data.shape)

#Preprocess the data to center (set mean to 0) and scale it (set standard dev. to 1)
scaled_data = preprocessing.scale(data.T)
	#Transpose because rows have to be the samples, not the columns
	#Could have used 'StandardScaler().fit_transform(data.T)' instead

#Generate the PCA
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

#Generate Scree Plot to see which PCA Components are important
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = ['PC' + str(i) for i in range(1, len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel("Percentage of Explained Variance")
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

#Transform the PCA Data into a Pandas Dataframe
pca_df = pd.DataFrame(pca_data, index=[*wt, *ko], columns=labels)

#Plot the PCA Data
plt.scatter(pca_df.PC1,pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))

for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

plt.show()

#Check Loading Scores
loading_scores = pd.Series(pca.components_[0], index=genes)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
top_10_genes = sorted_loading_scores[0:10].index.values
print(loading_scores[top_10_genes])

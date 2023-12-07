
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

iris=load_iris()
samples=iris.data
model=KMeans(n_clusters=3)
model.fit(samples)
labels=model.predict(samples)
print(labels)

# plotting sepal length and petal length
xs=samples[:,0]
ys=samples[:,2]
plt.scatter(xs,ys,c=labels)
centroids=model.cluster_centers_
centroids_x=centroids[:,0]
centroids_y=centroids[:,2]
plt.scatter(centroids_x,centroids_y,marker='D',s=50)
plt.show()

wine = pd.read_csv(r'C:\Users\Masum\Downloads\wine.csv')
wine.head()

wine.columns=['Class','Alcohol','Malic_acid','Ash','Alcalinity_of_ash','Magnesium','Total_phenols','Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue','OD280','Proline']
wine.head()

wine_class=wine['Class']
wine.drop('Class',axis=1,inplace=True)

model=KMeans(n_clusters=3)
labels=model.fit_predict(wine)

df=pd.DataFrame({'labels':labels , 'class':wine_class})
ct=pd.crosstab(df['labels'],df['class'])
ct

xs=wine.loc[:,'OD280']
ys=wine.loc[:,'Proline']
plt.scatter(xs,ys,c=labels)

wine.var(axis=0)

# Variance comparison between Proline and OD280
plt.scatter(wine['OD280'],wine['Proline'])
plt.xlim(-400,max(wine['Proline']))
plt.xlabel('OD280')
plt.ylabel('Proline')
plt.show()

"""Applying Standard Scaler (then KMeans in sklearn Pipeline)"""

#Applying Standard Scaler (then KMeans in sklearn Pipeline)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scaler=StandardScaler()
kmeans=KMeans(n_clusters=3)
pipeline=make_pipeline(scaler,kmeans)
pipeline.fit(wine)
labels=pipeline.predict(wine)

df=pd.DataFrame({'labels':labels,'class':wine_class   })
ct=pd.crosstab(df['labels'],df['class'])
ct
# After scaling we get tight clusters

xs=wine.loc[:,'OD280']
ys=wine.loc[:,'Proline']
plt.scatter(xs,ys,c=labels)

scaled_wine=scaler.fit_transform(wine)
scaled_wine.var(axis=0)

scaled_wine=pd.DataFrame(scaled_wine)
scaled_wine.columns=['Alcohol','Malic_acid','Ash','Alcalinity_of_ash','Magnesium','Total_phenols','Flavanoids','Nonflavanoid_phenols','Proanthocyanins','Color_intensity','Hue','OD280','Proline']

lables=KMeans(n_clusters=3).fit_predict(scaled_wine)
xs=scaled_wine.loc[:,'OD280']
ys=scaled_wine.loc[:,'Proline']
plt.scatter(xs,ys,c=labels)

"""Visualization with hierarchical clustering and t-SNE
Hierarchical Clustering
"""

from scipy.cluster.hierarchy import linkage,dendrogram , fcluster

from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
iris=load_iris()
data=iris.data
species=iris.target

import seaborn as sns
model=TSNE(learning_rate=100)
transformed=model.fit_transform(data)
xs=transformed[:,0]
ys=transformed[:,1]
plt.scatter(xs,ys,c=species)
#plt.legend(species)
#sns.scatterplot(xs,ys,hue=species)
plt.show()

# WINE DATA
# Samples contain two wine features
samples=wine[['Total_phenols','OD280']]
from sklearn.decomposition import PCA
model=PCA()
model.fit(samples)
transformed=model.transform(samples)
print(model.components_)
transformed[:10]


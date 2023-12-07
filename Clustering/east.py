# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:56:22 2023

@author: Masum
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score




# Load the data
data = pd.read_csv(r'C:\Users\Masum\Downloads\EastWestAirlines.xlsx')


# Dropping irrelevant columns for clustering
data_clustering = data.drop(['ID', 'Award'], axis=1)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_clustering)

# Hierarchical clustering
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    import scipy.cluster.hierarchy as sch
    sch.dendrogram(linkage_matrix, **kwargs)

plt.figure(figsize=(10, 5))
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(data_scaled)
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(model, truncate_mode='level', p=5)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# K-means clustering
# Elbow Method to determine the optimal number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Based on the elbow method, let's choose the optimal k and perform K-means clustering
k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
kmeans.fit(data_scaled)
data['KMeans_Cluster'] = kmeans.labels_

# DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(data_scaled)
data['DBSCAN_Cluster'] = dbscan.labels_

# Inferences from the clusters
# Analyze clusters by their mean values
cluster_means = data.groupby('KMeans_Cluster').mean()
print(cluster_means)

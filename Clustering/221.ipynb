# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:05:59 2023

@author: Masum
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# File path to your crimedata.csv file
file_path = 'C:\\Users\\Masum\\Downloads\\crime_data.csv'



# Load the data from the CSV file
crime_df = pd.read_csv(file_path)

# Display the first few rows of the dataset to ensure it's loaded properly
print(crime_df.head())

# Extracting necessary columns for clustering
crime_data = crime_df[['Murder', 'Assault', 'UrbanPop', 'Rape']]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(crime_data)

# Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
hc_clusters = hc.fit_predict(scaled_data)

# K-means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_clusters = kmeans.fit_predict(scaled_data)

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.8, min_samples=2)
dbscan_clusters = dbscan.fit_predict(scaled_data)

# Add cluster labels to the original dataframe
crime_df['HC_Cluster'] = hc_clusters
crime_df['KMeans_Cluster'] = kmeans_clusters
crime_df['DBSCAN_Cluster'] = dbscan_clusters

# Plotting the clusters (2D visualization, considering Murder and Assault)
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.title('Hierarchical Clustering')
plt.scatter(crime_df['Murder'], crime_df['Assault'], c=hc_clusters, cmap='viridis')
plt.xlabel('Murder')
plt.ylabel('Assault')

plt.subplot(132)
plt.title('K-means Clustering')
plt.scatter(crime_df['Murder'], crime_df['Assault'], c=kmeans_clusters, cmap='viridis')
plt.xlabel('Murder')
plt.ylabel('Assault')

plt.subplot(133)
plt.title('DBSCAN Clustering')
plt.scatter(crime_df['Murder'], crime_df['Assault'], c=dbscan_clusters, cmap='viridis')
plt.xlabel('Murder')
plt.ylabel('Assault')

plt.tight_layout()
plt.show()

# Displaying the number of clusters formed by each method
print("Hierarchical Clustering - Number of clusters:", len(set(hc_clusters)))
print("K-means Clustering - Number of clusters:", len(set(kmeans_clusters)))
print("DBSCAN Clustering - Number of clusters:", len(set(dbscan_clusters)))

# Displaying the cluster labels for each data point
print("\nHierarchical Clustering Labels:")
print(hc_clusters)
print("\nK-means Clustering Labels:")
print(kmeans_clusters)
print("\nDBSCAN Clustering Labels:")
print(dbscan_clusters)





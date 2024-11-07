# Implement K-Means clustering/ hierarchical clustering on sales_data_sample.csv dataset.
# Determine the number of clusters using the elbow method.
# Dataset link : https://www.kaggle.com/datasets/kyanyoga/sample-sales-data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score

# Load the dataset
df = pd.read_csv('sales_data_sample.csv')

# Check the first few rows of the dataset
print(df.head())

# Step 1: Preprocess the data
# Let's assume we are clustering based on 'Sales' and 'Profit' columns, as an example.
df = df[['Sales', 'Profit']]

# Handle missing values by filling them with the mean (or you can drop rows with NaN)
df.fillna(df.mean(), inplace=True)

# Step 2: Feature Scaling (important for clustering)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Step 3: Determine the optimal number of clusters using the Elbow Method for K-Means
# We will calculate the sum of squared distances for different values of k
wcss = []  # Within-cluster sum of squares (WCSS)

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow method to find the optimal k
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within-cluster Sum of Squares)')
plt.show()

# From the plot, identify the 'elbow' point for optimal k (e.g., k=3 or k=4).

# Step 4: Apply K-Means with the chosen number of clusters (let's assume k=3)
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(df_scaled)

# Step 5: Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(df_scaled[y_kmeans == 0, 0], df_scaled[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(df_scaled[y_kmeans == 1, 0], df_scaled[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(df_scaled[y_kmeans == 2, 0], df_scaled[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')

plt.title('K-Means Clustering')
plt.xlabel('Sales')
plt.ylabel('Profit')
plt.legend()
plt.show()

# Step 6: Evaluate the Clusters using Silhouette Score
silhouette_avg = silhouette_score(df_scaled, y_kmeans)
print(f'Silhouette Score for K-Means: {silhouette_avg:.4f}')

# Step 7: Hierarchical Clustering (Agglomerative)
Z = linkage(df_scaled, 'ward')  # 'ward' minimizes the variance of merged clusters

# Plot the Dendrogram
plt.figure(figsize=(10, 8))
dendrogram(Z)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# You can choose the number of clusters by cutting the dendrogram at a certain level

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Fix for Joblib CPU issue
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Manually set number of CPU cores

# Load the dataset
file_path = "G:/TU Dresden/Thesis/Sample_data_for_machine_learning/Run7/T35_46_In160_300dp_2100n_100d/piston.txt"
df = pd.read_csv(file_path, delimiter="\t")

# Debug: Print first few rows to understand the data
print("Initial Data Sample:")
print(df.head())

# Drop non-numeric or identifier columns if necessary
df_numeric = df.iloc[:, 3:]

# Debug: Print summary of numeric data
print("\nSummary of Numeric Data:")
print(df_numeric.describe())

# Handle missing values
df_numeric.fillna(df_numeric.mean(), inplace=True)

# Debug: Check for any remaining missing values
print("\nMissing Values in Data:")
print(df_numeric.isnull().sum())

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# Debug: Check shape of scaled data
print("\nShape of Scaled Data:", X_scaled.shape)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Debug: Print explained variance ratio
print("\nExplained Variance Ratio by PCA Components:")
print(pca.explained_variance_ratio_)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
K_range = range(2, 11)  # Checking K values from 2 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

# # Plot the Elbow Method results
# plt.figure(figsize=(8, 5))
# plt.plot(K_range, inertia, marker='o', linestyle='-')
# plt.xlabel('Number of Clusters (K)')
# plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
# plt.title('Elbow Method for Optimal K')
# plt.grid(True)
# plt.show()
#
# # Perform K-Means clustering with optimal K (assuming 4 from previous analysis)
# optimal_k = 4
# kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
# kmeans_labels = kmeans.fit_predict(X_pca)
#
# # Debug: Print first 10 cluster labels
# print("\nFirst 10 Cluster Labels from K-Means:")
# print(kmeans_labels[:10])
#
# # Visualizing clusters using PCA (2D)
# plt.figure(figsize=(8, 6))
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6, edgecolors='k')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.title('K-Means Clustering (2D PCA Projection)')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # Visualizing clusters using PCA (3D)
# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=kmeans_labels, cmap='viridis', alpha=0.6, edgecolors='k')
# ax.set_xlabel('PCA Component 1')
# ax.set_ylabel('PCA Component 2')
# ax.set_zlabel('PCA Component 3')
# ax.set_title('K-Means Clustering (3D PCA Projection)')
# plt.show()

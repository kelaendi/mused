# Make all imports I could maybe need
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from collections import deque
from PIL import Image
from PIL.ExifTags import TAGS
# import cv2
# import exifread
# import requests

def generate_random_multimodal_data(N, seed):
    np.random.seed(seed)
    time_data = np.random.rand(N, N//10)
    text_data = np.random.rand(N, N//2)
    image_data = np.random.rand(N, N)
    social_data = np.random.rand(N, N//5)
    location_data = np.random.rand(N, N)
    return [time_data, text_data, image_data, social_data, location_data]

def create_adjacency_matrix(data, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    matrix = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in indices[i]: #i.e. for each of the k nearest neighbors
            matrix[i, j] = 1
            matrix[j, i] = 1
    return matrix

def fuse_reduce_and_cluster(N, data, k_neighbors, reduced_dim, n_clusters, seed):
    # Create an adjacency matrix for each modality
    adjacency_matrices = []
    for modality in data:
        adjacency_matrices.append(create_adjacency_matrix(modality, k_neighbors))

    # Construct multimodal adjacency matrix with logical OR
    fused_matrix = np.zeros((N, N))
    for matrix in adjacency_matrices:
        fused_matrix = np.logical_or(fused_matrix, matrix)
    fused_matrix = fused_matrix.astype(int) # int instead of boolean
    
    # Apply SVD to reduce dimensionality
    svd = TruncatedSVD(n_components=reduced_dim)
    reduced_matrix = svd.fit_transform(fused_matrix)
    
    # Cluster with K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    clusters = kmeans.fit_predict(reduced_matrix)

    return reduced_matrix, clusters

def visualize_clusters(reduced_matrix, clusters, plot_name="cluster_vis", save_path="plots/"):
    # Ensure the sava path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Reduce to 2D
    svd_vis = TruncatedSVD(n_components=2)
    matrix_vis = svd_vis.fit_transform(reduced_matrix)
    plt.scatter(matrix_vis[:,0], matrix_vis[:,1], c=clusters)
    plt.title(f"Cluster Visualization {plot_name}")
    plt.xlabel('x')
    plt.ylabel('y')

    plot_filename = os.path.join(save_path, plot_name)

    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")


def run():
    # Set parameters to tweak
    N = 100
    seed = 123
    k_neighbors = 10 # sqrt(N)
    reduced_dim = 20 # cumulative variance plot? set so dim explains 90%
    n_clusters = 2 # see elbow method
    plot_name = f"N={N},k={k_neighbors},seed={seed}"

    random_data = generate_random_multimodal_data(N, seed)
    reduced_data, clusters = fuse_reduce_and_cluster(N, random_data, k_neighbors, reduced_dim, n_clusters, seed)
    visualize_clusters(reduced_data, clusters, plot_name)
    

if __name__ == "__main__":
    run()

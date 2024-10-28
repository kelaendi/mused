import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from swfd import SeqBasedSWFD
from collections import deque
import math

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

def fuse_reduce_and_cluster(window_data, k_neighbors, reduced_dim, n_clusters, seed):
    # Create an adjacency matrix for each modality
    adjacency_matrices = []
    for modality in window_data:
        adjacency_matrices.append(create_adjacency_matrix(modality, k_neighbors))

    # Construct multimodal adjacency matrix with logical OR
    fused_matrix = np.zeros((len(window_data[0]), len(window_data[0])))
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

def process_streaming_data(data, window_size, k_neighbors, reduced_dim, n_clusters, seed, save_path="logs/"):
    
    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log_name = f"seed={seed},N={len(data[0])},window_size={window_size},k={k_neighbors}.txt"

    log_file = os.path.join(save_path, log_name)

    with open(log_file, "w") as f_log:
        f_log.write("Window_End_Index\tSilhouette_Score\tCenter_Shift\tAnomaly_Detected\n")
        
    swfd = SeqBasedSWFD(window_size, R=1, d=len(data[0][0]), sketch_dim=reduced_dim)

    # Initialize sliding window as a deque, ensuring old data points get removed as new ones are added
    window = deque(maxlen=window_size)
    all_clusters = []
    prev_centers = None #for keeping track of cluster center shifts

    # Simulate streaming data
    for i in range(len(data[0])):
        # Collect a single data point from each modality
        data_point = [modality[i:i+1] for modality in data]  
        window.append(data_point)

        # Only process once we have a full window
        if len(window) == window_size:
            window_data = [np.concatenate([point[mod] for point in window], axis=0) for mod in range(len(data))]

            # Reduce and cluster on the window
            reduced_data, clusters = fuse_reduce_and_cluster(window_data, k_neighbors, reduced_dim, n_clusters, seed)
            
            # Update the sketch with reduced data
            swfd.fit(reduced_data)
            all_clusters.append(clusters)

            # Calculate metrics
            silhouette_avg = silhouette_score(reduced_data, clusters) if len(set(clusters)) > 1 else -1
            centers = np.array([reduced_data[clusters == c].mean(axis=0) for c in range(n_clusters)])
            center_shift = np.linalg.norm(centers - prev_centers) if prev_centers is not None else 0
            prev_centers = centers

            # Flag anomaly if silhouette score is low or shift is large
            # TODO how to best set this thresholds?
            anomaly_detected = silhouette_avg < 0.2 or center_shift > 1.0
            anomaly_flag = "Yes" if anomaly_detected else "No"

            # Log the metrics to the file
            with open(log_file, "a") as log:
                log.write(f"{i}\t{silhouette_avg:.3f}\t{center_shift:.3f}\t{anomaly_flag}\n")

            if (i%window_size==0 or i==len(data[0])-1):
                plot_name = f"seed={seed},N={len(data[0])},window_size={window_size},k={k_neighbors},i={i}"
                visualize_clusters(reduced_data, clusters, plot_name)
                # B_t, _, _, delta = swfd.get()
                # print(B_t)

def run():
    # Set parameters to tweak
    N = 1000
    seed = 123
    window_size = 200  # Size of sliding window
    k_neighbors = math.floor(math.sqrt(window_size))
    reduced_dim = window_size//20 # cumulative variance plot? set so dim explains 90%
    n_clusters = 2 #normal vs anomalous

    random_data = generate_random_multimodal_data(N, seed)
    process_streaming_data(random_data, window_size, k_neighbors, reduced_dim, n_clusters, seed)

    print(f"Finished processing for seed={seed},N={N},window_size={window_size},k={k_neighbors}")

if __name__ == "__main__":
    run()

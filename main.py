import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from swfd import SeqBasedSWFD
from collections import deque
import math
import scipy.io
import time

def create_adjacency_matrix(data, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    matrix = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in indices[i]: #i.e. for each of the k nearest neighbors
            matrix[i, j] = 1
            matrix[j, i] = 1
    return matrix

def fuse_matrices(matrices):
    fused_matrix = matrices[0].copy()
    for matrix in matrices[1:]:
        fused_matrix = np.logical_or(fused_matrix, matrix)
        fused_matrix = fused_matrix.astype(int)
    print(f"Fused matrix shape: {fused_matrix.shape}")
    print(f"Non-zero entries in fused matrix: {np.count_nonzero(fused_matrix)}")
    return fused_matrix


def reduce_and_cluster(fused_matrix, reduced_dim, n_clusters, seed):
    # Apply SVD to reduce dimensionality
    svd_dim = min(reduced_dim, fused_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=svd_dim)
    reduced_matrix = svd.fit_transform(fused_matrix)

    print(f"Reduced matrix shape: {reduced_matrix.shape}")
    
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

def process_streaming_data(data_modalities, window_size, k_neighbors, reduced_dim, n_clusters, seed, save_path="logs/"):

    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log_name = f"seed={seed},window_size={window_size},k={k_neighbors}.txt"

    log_file = os.path.join(save_path, log_name)

    with open(log_file, "w") as f_log:
        f_log.write("max_error\tavg_error\tavg_update_time\tavg_query_time\tmax_size\tWindow_End_Index\tSilhouette_Score\tCenter_Shift\tAnomaly_Detected\n")

    sketches = []
    all_clusters = []
    prev_centers = None #for keeping track of cluster center shifts
    # Initialize sliding window as a deque, ensuring old data points get removed as new ones are added
    window = deque(maxlen=window_size)

    # Initialize metrics
    max_error = 0
    sum_error = 0
    update_time_total = 0
    query_time_total = 0
    query_count = 0
    max_size = 0

    for modality in data_modalities:
        # Determine R dynamically
        max_norm = np.max(np.linalg.norm(modality[0], axis=1)**2)
        sketches.append(SeqBasedSWFD(window_size, R=max_norm, d=len(modality[0][0]), sketch_dim=reduced_dim))

    print("Simulate streaming data, updating sketch with each arriving datapont")
    for i in range(len(data_modalities[0][0])):
        # Collect a single data point from each modality
        data_point = [modality[0][i:i+1] for modality in data_modalities]  

        window.append(data_point)
        for j, sketch in enumerate(sketches):
            start_time = time.process_time_ns()
            sketch.fit(np.array(data_point[j]))
            end_time = time.process_time_ns()
            update_time_total += (end_time - start_time) / 1e6 #in ms

        # Only process once we have a full window
        if len(window) == window_size and (i+1)%window_size==0:
            print(f"i={i}")
            adjacency_matrices = []
            for sketch in sketches:
                start_time = time.process_time_ns()
                B_t, _, _, _ = sketch.get()
                end_time = time.process_time_ns()
                query_time_total += (end_time - start_time) / 1e6  # Convert to ms
                query_count += 1

                # Error metrics
                A_w = np.concatenate([point[sketches.index(sketch)] for point in window])
                A_w_norm = np.linalg.norm(A_w) ** 2
                B_w_norm = np.linalg.norm(B_t) ** 2
                relative_error = np.abs(A_w_norm - B_w_norm) / A_w_norm
                max_error = max(max_error, relative_error)
                sum_error += relative_error

                adjacency_matrices.append(create_adjacency_matrix(B_t, min(k_neighbors, B_t.shape[0]-1)))

            fused_matrix = fuse_matrices(adjacency_matrices)

            # Reduce and cluster on the window
            print("Reduce and cluster...")
            reduced_data, clusters = reduce_and_cluster(fused_matrix, reduced_dim, n_clusters, seed)
            
            all_clusters.append(clusters)

            # Calculate metrics
            print("Calculate metrics...")
            silhouette_avg = silhouette_score(reduced_data, clusters) if len(set(clusters)) > 1 else -1
            centers = np.array([reduced_data[clusters == c].mean(axis=0) for c in range(n_clusters)])
            center_shift = np.linalg.norm(centers - prev_centers) if prev_centers is not None else 0
            prev_centers = centers

            # Flag anomaly if silhouette score is low or shift is large
            # TODO how to best set this thresholds?
            anomaly_detected = silhouette_avg < 0.2 or center_shift > 1.0
            anomaly_flag = "Yes" if anomaly_detected else "No"

            avg_error = sum_error / query_count if query_count > 0 else 0
            avg_update_time = update_time_total / i if i > 0 else 0
            avg_query_time = query_time_total / query_count if query_count > 0 else 0
            max_size = max(max_size, sketch.get_size())

            # Log the metrics to the file
            with open(log_file, "a") as log:
                log.write(f"{i}\t{silhouette_avg:.3f}\t{center_shift:.3f}\t{anomaly_flag}\t{max_error:.6f}\t{avg_error:.6f}\t{avg_update_time:.6f}\t{avg_query_time:.6f}\t{max_size}\n")

            if (i==len(data_modalities[0][0])-1): #i%window_size==0 or 
                plot_name = f"seed={seed},N={len(data_modalities[0][0])},window_size={window_size},k={k_neighbors},i={i}"
                visualize_clusters(reduced_data, clusters, plot_name)
                # B_t, _, _, delta = swfd.get()
                # print(B_t)

def load_dataset(file_path):
    data = scipy.io.loadmat(file_path)["A"]
    return [data.astype(np.float64)]

def run():
    # Set parameters to tweak
    synthetic_path = "swfd/dataset/synthetic_n=500000,m=10,d=300,zeta=10.mat"

    # Parameters
    N = 500000
    window_size = 100000
    reduced_dim = 8
    
    # N = 1000
    seed = 123
    # window_size = 200  # Size of sliding window
    k_neighbors = 8 #50 #math.floor(math.sqrt(window_size))
    # reduced_dim = window_size//20 # cumulative variance plot? set so dim explains 90%
    n_clusters = 2 #normal vs anomalous

    # random_data = generate_random_multimodal_data(N, seed)

    print("Loading synthetic dataset...")
    synthetic_data = load_dataset(synthetic_path)

    # process_streaming_data(random_data, window_size, k_neighbors, reduced_dim, n_clusters, seed)

    print("Processing streaming data...")
    process_streaming_data([synthetic_data, synthetic_data], window_size, k_neighbors, reduced_dim, n_clusters, seed)

    print(f"Finished processing for seed={seed},N={N},window_size={window_size},k={k_neighbors}")

if __name__ == "__main__":
    run()

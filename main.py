import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from swfd import SeqBasedSWFD
from collections import deque
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
    # print(f"Fused matrix shape: {fused_matrix.shape}")
    # print(f"Non-zero entries in fused matrix: {np.count_nonzero(fused_matrix)}")
    return fused_matrix

def perform_svd_reduction(matrix, reduced_dim, seed):
    # Apply SVD to reduce dimensionality
    svd_dim = min(reduced_dim, matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=svd_dim)
    return svd.fit_transform(matrix)

def perform_clustering(matrix, n_clusters, seed):
    # Cluster with K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    clusters = kmeans.fit_predict(matrix)
    return clusters

def visualize_results(metrics, metric_name, save_path="plots/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(10, 6))
    for approach, values in metrics.items():
        plt.plot(values["window_indices"], values[metric_name], label=approach)
    plt.title(f"{metric_name} - Approach Comparison")
    plt.xlabel("Window End Index")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, f"{metric_name}_comparison.png"))
    plt.close()

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

def process_streaming_data(data_modalities, window_size, k_neighbors, reduced_dim, n_clusters, seed, approach):

    total_size = len(data_modalities[0][0])
    window = deque(maxlen=window_size)
    results = {"window_indices": [], "silhouette": [], "norm_delta": [], "processing_time": []}
    sketches = []
    added_processing_time = 0

    # Setup SWFD first if that's the approach chosen
    if approach == "SWFD_first":
        start_time = time.process_time_ns()
        for modality in data_modalities:
            # Determine R dynamically
            max_norm = np.max(np.linalg.norm(modality[0], axis=1)**2)
            sketches.append(SeqBasedSWFD(window_size, R=max_norm, d=len(modality[0][0]), sketch_dim=reduced_dim))
        end_time = time.process_time_ns()
        added_processing_time += (end_time - start_time)

    #Simulate streaming data, updating sketch with each arriving datapoint
    for i in range(total_size):
        # Collect a single data point from each modality
        data_point = [modality[0][i:i+1] for modality in data_modalities]  
        window.append(data_point)

        if approach == "SWFD_first":
            start_time = time.process_time_ns()
            for j, sketch in enumerate(sketches):
                sketch.fit(np.array(data_point[j]))
            end_time = time.process_time_ns()
            added_processing_time += (end_time - start_time)

        # Only process once we have a full window
        if len(window) == window_size and (i + 1) % window_size == 0:
            print(f"i={i}")
            window_start_time= time.process_time_ns()

            if approach == "naive":
                # Fuse data
                adjacency_matrices = []
                for m_index, modality in enumerate(data_modalities):
                    A_w = np.concatenate([point[m_index] for point in window], axis=0)
                    adjacency_matrices.append(create_adjacency_matrix(A_w, min(k_neighbors, A_w.shape[0]-1)))
                fused_matrix = fuse_matrices(adjacency_matrices)

                # Reduce with SWFD sketching
                max_norm = np.max(np.linalg.norm(fused_matrix, axis=1)**2)
                swfd = SeqBasedSWFD(window_size, R=max_norm, d=fused_matrix.shape[1], sketch_dim=reduced_dim)
                for row in fused_matrix:
                    swfd.fit(row[np.newaxis, :])
                reduced_matrix, _, _, _ = swfd.get()
            
            elif approach == "SVD":
                # Fuse data
                adjacency_matrices = []
                for m_index, modality in enumerate(data_modalities):
                    A_w = np.concatenate([point[m_index] for point in window], axis=0)
                    adjacency_matrices.append(create_adjacency_matrix(A_w, min(k_neighbors, A_w.shape[0]-1)))
                fused_matrix = fuse_matrices(adjacency_matrices)

                # Reduce with SVD
                reduced_matrix = perform_svd_reduction(fused_matrix, reduced_dim, seed)
            
            elif approach == "SWFD_first":
                # First reduced via SWFD then fused
                adjacency_matrices = []
                for sketch in sketches:
                    B_t, _, _, _ = sketch.get()
                    A_w = np.concatenate([point[sketches.index(sketch)] for point in window])
                    adjacency_matrices.append(create_adjacency_matrix(B_t, min(k_neighbors, B_t.shape[0]-1)))
                reduced_matrix = fuse_matrices(adjacency_matrices)
            else:
                reduced_matrix = fused_matrix

            # Clustering
            clusters = perform_clustering(reduced_matrix, n_clusters, seed)
        
            window_end_time= time.process_time_ns()

            # Evaluation metrics
            silhouette_avg = silhouette_score(reduced_matrix, clusters) if len(set(clusters)) > 1 else -1
            processing_time = (window_end_time - window_start_time + added_processing_time) / 1e9
            added_processing_time = 0

            if approach == "SWFD_first":
                # Get fused and not yet reduced matrix for norm delta comparison
                adjacency_matrices = []
                for m_index, modality in enumerate(data_modalities):
                    A_w = np.concatenate([point[m_index] for point in window], axis=0)
                    adjacency_matrices.append(create_adjacency_matrix(A_w, min(k_neighbors, A_w.shape[0]-1)))

                fused_not_reduced = fuse_matrices(adjacency_matrices)
                fused_not_reduced_norm = np.linalg.norm(fused_not_reduced, 'fro')
                fused_and_reduced_norm = np.linalg.norm(reduced_matrix, 'fro')
                norm_delta = abs(fused_not_reduced_norm - fused_and_reduced_norm) / fused_not_reduced_norm
            else:
                fused_norm = np.linalg.norm(fused_matrix, 'fro') #frobinius
                reduced_norm = np.linalg.norm(reduced_matrix, 'fro')
                norm_delta = abs(fused_norm - reduced_norm) / fused_norm 

             # Record metrics
            results["window_indices"].append(i)
            results["silhouette"].append(silhouette_avg)
            results["norm_delta"].append(norm_delta)
            results["processing_time"].append(processing_time)

    return results

# def log_metrics(results,total_size,window_size,reduced_dim,k_neighbors,seed,save_path="logs/"):
#     # Ensure the save path exists
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     log_name = f"svd,seed={seed},n={total_size},window_size={window_size},reduced_dim={reduced_dim},k={k_neighbors}.txt"

#     log_file = os.path.join(save_path, log_name)

#     with open(log_file, "w") as f_log:
#         # f_log.write("max_error\tavg_error\tavg_update_time\tavg_query_time\tmax_size\tWindow_End_Index\tSilhouette_Score\tCenter_Shift\tAnomaly_Detected\n")
#         f_log.write("Window_End_Index\tSilhouette_Score\tCenter_Shift\tAnomaly_Detected\tNorm_Delta\tProcessing_Time\n")

#     # Log the metrics to the file
#     with open(log_file, "a") as log:
#         # log.write(f"{max_error:.6f}\t{avg_error:.6f}\t{avg_update_time:.6f}\t{avg_query_time:.6f}\t{max_size}\t{i}\t{silhouette_avg:.3f}\t{center_shift:.3f}\t{anomaly_flag}\n")
#         log.write(f"{i}\t{results["silhouette"]:.3f}\t{results["center_shift"]:.3f}\t{results["norm_delta"]}\t{results["processing_time"]}\n")

def load_dataset(file_path, subset_size=None):
    data = scipy.io.loadmat(file_path)["A"]
    if subset_size is not None and subset_size>0 and subset_size<len(data):
        data = data[:subset_size]
    return [data.astype(np.float64)]

def run():
    # Set parameters to tweak
    synthetic_path = "swfd/dataset/synthetic_n=500000,m=10,d=300,zeta=10.mat"

    # Parameters
    seed = 0
    subset_size = 5000 # total_size = 500000
    window_size = 1000
    reduced_dim = 80
    k_neighbors = 50
    n_clusters = 2 #normal vs anomalous

    print("Loading dataset...")
    synthetic_data = load_dataset(synthetic_path, subset_size)
    
    approaches = ["naive", "SVD", "SWFD_first"]
    metrics = {}

    start_total_time = time.process_time_ns()

    for approach in approaches:
        print(f"Start processing with {approach} approach...")

        start_approach_time = time.process_time_ns()
        metrics[approach] = process_streaming_data(
            [synthetic_data, synthetic_data],
            window_size,
            k_neighbors,
            reduced_dim,
            n_clusters,
            seed,
            approach,
        )
        end_approach_time = time.process_time_ns()
        approach_processing_time = (end_approach_time - start_approach_time) / 1e9
        print(f"Processed with {approach} approach for {approach_processing_time} seconds")

    # Visualize results
    for metric_name in ["silhouette", "norm_delta", "processing_time"]:
        visualize_results(metrics, metric_name)
    
    end_total_time = time.process_time_ns()

    total_processing_time = (end_total_time - start_total_time) / 1e9

    print(f"Finished all processing for seed={seed},N={subset_size},window_size={window_size},k={k_neighbors}")
    print(f"Total processing time: {total_processing_time} seconds")

if __name__ == "__main__":
    run()

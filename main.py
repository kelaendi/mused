import os
import time
import scipy.io
import numpy as np
from collections import deque
from swfd import SeqBasedSWFD
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import (normalized_mutual_info_score, f1_score as sklearn_f1_score, precision_score, recall_score, accuracy_score,
                             silhouette_score as sklearn_silhouette_score, davies_bouldin_score, average_precision_score, roc_auc_score, mean_absolute_error)

def create_adjacency_matrix(data, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    matrix = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in indices[i]:  # i.e. for each of the k nearest neighbors
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

def compute_dunn_index(data, labels):
    unique_labels = np.unique(labels)
    centroids = [data[labels == c].mean(axis=0) for c in unique_labels]
    intra_cluster_dists = [np.mean(np.linalg.norm(
        data[labels == c] - centroids[i], axis=1)) for i, c in enumerate(unique_labels)]
    inter_cluster_dists = [np.linalg.norm(
        c1 - c2) for i, c1 in enumerate(centroids) for c2 in centroids[i+1:]]
    return min(inter_cluster_dists) / max(intra_cluster_dists)

def visualize_results(metrics, save_path="plots/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    metric_names = list(next(iter(metrics.values())).keys())
    
    for metric_name in metric_names:
        if metric_name == "window_indices":
            continue

        plt.figure(figsize=(10, 6))
        for approach, values in metrics.items():
            if metric_name in values:
                plt.plot(values["window_indices"], values[metric_name], label=approach)
        metric_label = metric_name.replace('_', ' ').capitalize()
        plt.title(f"{metric_label} - Approach Comparison")
        plt.xlabel("Window End Index")
        plt.ylabel(metric_label)
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
    plt.scatter(matrix_vis[:, 0], matrix_vis[:, 1], c=clusters)
    plt.title(f"Cluster Visualization {plot_name}")
    plt.xlabel('x')
    plt.ylabel('y')

    plot_filename = os.path.join(save_path, plot_name)

    plt.savefig(plot_filename)

def process_streaming_data(data_modalities, window_size, k_neighbors, reduced_dim, n_clusters, seed, approach):

    total_size = len(data_modalities[0][0])
    window = deque(maxlen=window_size)
    results = {
    "window_indices": [],
    "nmi_score": [],
    "f1_score": [],
    "precision": [],
    "recall": [],
    "accuracy": [],
    "db_index": [],
    "dunn_index": [],
    "map_score": [],
    "auc_score": [],
    "mae": [],
    "silhouette_score": [],
    "norm_delta": [],
    "processing_time": [],
    }
    sketches = []
    added_processing_time = 0
    # TODO replace the provisory true labels
    complete_true_labels = np.random.randint(0, n_clusters, size=(total_size,))


    # Setup SWFD first if that's the approach chosen
    if approach == "SWFD_first":
        start_time = time.process_time_ns()
        for modality in data_modalities:
            # Determine R dynamically
            max_norm = np.max(np.linalg.norm(modality[0], axis=1)**2)
            sketches.append(SeqBasedSWFD(window_size, R=max_norm, d=len(modality[0][0]), sketch_dim=reduced_dim))
        end_time = time.process_time_ns()
        added_processing_time += (end_time - start_time)

    # Simulate streaming data, updating sketch with each arriving datapoint
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
            window_start_time = time.process_time_ns()

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

            window_end_time = time.process_time_ns()

            true_labels = complete_true_labels[i - len(reduced_matrix) + 1:i + 1]
            print(f"true_labels.shape: {true_labels.shape}, clusters.shape: {clusters.shape}")


            # Generate prediction scores based on distances to cluster centers
            cluster_centers = np.array([reduced_matrix[clusters == c].mean(axis=0) for c in range(n_clusters)])
            predicted_scores = np.linalg.norm(reduced_matrix[:, None, :] - cluster_centers[clusters], axis=2).diagonal()

            # Evaluation metrics
            nmi_score = normalized_mutual_info_score(true_labels, clusters)
            print(f"nmi={nmi_score}")
            f1_score = sklearn_f1_score(true_labels, clusters, average='weighted')
            print(f"f1={f1_score}")
            precision = precision_score(true_labels, clusters, average='weighted')
            print(f"precision={precision}")
            recall = recall_score(true_labels, clusters, average='weighted')
            print(f"recall={recall}")
            accuracy = accuracy_score(true_labels, clusters)
            print(f"accuracy={accuracy}")
            db_index = davies_bouldin_score(reduced_matrix, clusters)
            print(f"db_index={db_index}")
            dunn_index = compute_dunn_index(reduced_matrix, clusters)
            print(f"dunn_index={dunn_index}")
            map_score = average_precision_score(true_labels, predicted_scores)
            print(f"map={map_score}")
            auc_score = roc_auc_score(true_labels, predicted_scores)
            print(f"auc={auc_score}")
            mae = mean_absolute_error(true_labels, clusters)
            print(f"mae={mae}")
            silhouette_score = sklearn_silhouette_score(reduced_matrix, clusters) if len(set(clusters)) > 1 else -1
            print(f"silhouette={silhouette_score}")
            processing_time = (window_end_time - window_start_time + added_processing_time) / 1e9
            print(f"processing_time={processing_time}")
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
                fused_norm = np.linalg.norm(fused_matrix, 'fro')  # frobinius
                reduced_norm = np.linalg.norm(reduced_matrix, 'fro')
                norm_delta = abs(fused_norm - reduced_norm) / fused_norm

            # Record metrics
            results["window_indices"].append(i)
            results["nmi_score"].append(nmi_score)
            results["f1_score"].append(f1_score)
            results["precision"].append(precision)
            results["recall"].append(recall)
            results["accuracy"].append(accuracy)
            results["db_index"].append(db_index)
            results["dunn_index"].append(dunn_index)
            results["map_score"].append(map_score)
            results["auc_score"].append(auc_score)
            results["mae"].append(mae)
            results["silhouette_score"].append(silhouette_score)
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
    if subset_size is not None and subset_size > 0 and subset_size < len(data):
        data = data[:subset_size]
    return [data.astype(np.float64)]


def run():
    # Set parameters to tweak
    synthetic_path = "swfd/dataset/synthetic_n=500000,m=10,d=300,zeta=10.mat"

    # Parameters
    seed = 0
    total_size = 500000
    subset_size = 5000
    window_size = 1000
    reduced_dim = 80
    k_neighbors = 50
    n_clusters = 2  # normal vs anomalous
    np.random.seed(seed)

    print("Loading dataset...")
    synthetic_data = load_dataset(synthetic_path, subset_size)
    # true_labels = np.random.randint(0, n_clusters, size=(total_size,))

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
        approach_processing_time = (
            end_approach_time - start_approach_time) / 1e9
        print(
            f"Processed with {approach} approach for {approach_processing_time} seconds")

    visualize_results(metrics)

    end_total_time = time.process_time_ns()

    total_processing_time = (end_total_time - start_total_time) / 1e9

    print(
        f"Finished all processing for seed={seed},N={subset_size},window_size={window_size},k={k_neighbors}")
    print(f"Total processing time: {total_processing_time} seconds")


if __name__ == "__main__":
    run()

import time
import numpy as np
import data_loader
from matrix_operations import create_adjacency_matrix, fuse_matrices, perform_clustering, perform_svd_reduction
import metrics_evaluation
import output_generation
from collections import deque
from swfd import SeqBasedSWFD

def process_streaming_data(results, data_modalities, window_size, k_neighbors, reduced_dim, n_clusters, seed, approach, complete_true_labels, step_window_ratio, noise_rate, label_mode, sorting):

    subset_size = len(data_modalities[0])
    print(f"subset_size = {subset_size}")

    total_start_time = time.time_ns()

    window = deque(maxlen=window_size)
    
    sketches = []

    # To store clusters and labels for the entire subset
    all_clusters = []
    all_true_labels = []

    # Setup SWFD first if that's the approach chosen
    if approach == "SWFD_first":
        for modality in data_modalities:
            print(f"modality.shape: {modality.shape}")

            if modality.ndim == 1:
                modality = modality.reshape(-1, 1)

            # Determine R dynamically
            max_norm = np.max(np.linalg.norm(modality, axis=1)**2)
            sketches.append(SeqBasedSWFD(window_size, R=max_norm, d=modality.shape[1], sketch_dim=reduced_dim))

    # Simulate streaming data, updating sketch with each arriving datapoint
    for i in range(subset_size):
        # Collect a single data point from each modality
        data_point = [modality[i:i+1] for modality in data_modalities]
        window.append(data_point)

        if approach == "SWFD_first":
            for j, sketch in enumerate(sketches):
                sketch.fit(data_point[j])

        # Only process once we have a full window
        if len(window) == window_size and (i + 1)*step_window_ratio % window_size == 0:
            print(f"i={i}")

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
                reduced_matrix = data_modalities

            # Clustering
            clusters = perform_clustering(reduced_matrix, n_clusters, seed)

            # Accumulate all clusters and true labels
            all_clusters.extend(clusters)
            true_labels = complete_true_labels[i - len(reduced_matrix) + 1:i + 1]
            all_true_labels.extend(true_labels)

    total_end_time = time.time_ns()

    # Compute metrics for the entire subset
    all_true_labels = np.array(all_true_labels)
    all_clusters = np.array(all_clusters)

    print(f"all_true_labels.shape: {all_true_labels.shape}, all_clusters.shape: {all_clusters.shape}")
    print(f"Unique labels: {np.unique(all_true_labels)}")

    # Compute metrics for the entire subset
    results = metrics_evaluation.compute_all_metrics(results, subset_size, noise_rate, label_mode, sorting, all_clusters, all_true_labels, total_end_time, total_start_time)

    return results

def run():
    # Parameters
    seed = 0
    subset_sizes = [1000, 2000, 4000, 8000]  # Example sizes
    noise_rates = [0, 25, 50, 95]
    # noise_rate = noise_rates[-1]
    max_subset_size = max(subset_sizes)
    window_size = 1000
    reduced_dim = 80
    k_neighbors = 50
    n_clusters = 0 #event related vs not #4  # technical, soccer, indignados
    step_window_ratio= 1
    label_modes = ["binary", "types", "all"]
    label_mode = label_modes[0]
    sorting = True
    np.random.seed(seed)

    # Decide on defaults for all independent variables, maybe 4000 subset, 50% noise, binary mode, non sorted

    print("Loading dataset...")
    match label_mode:
        case "binary":
            n_clusters = 2
            modalities, truth_labels = data_loader.load_sed2012_dataset(subset_size=max_subset_size, binary=True, event_types=True, sort_by_uploaded=sorting)
    
        case "types":
            n_clusters = 4
            modalities, truth_labels = data_loader.load_sed2012_dataset(subset_size=max_subset_size, binary=False, event_types=True, sort_by_uploaded=sorting)
    
        case _:
            n_clusters = 0
            modalities, truth_labels = data_loader.load_sed2012_dataset(subset_size=max_subset_size, binary=False, event_types=False, sort_by_uploaded=sorting)
    
    noise_rate = np.sum(truth_labels == 0) / len(truth_labels)
    details_string = f"_mode={label_mode},sorted={sorting},noise={noise_rate},window={window_size},subset={max_subset_size},k={k_neighbors},dim={reduced_dim}"

    approaches = [
        "naive", 
        "SVD", 
        # "SWFD_first"
        ]
    
    metrics = {}

    start_total_time = time.time_ns()

    for approach in approaches:
        print(f"Start processing with {approach} approach...")
        results, independent_variables = metrics_evaluation.get_initial_results()
        start_approach_time = time.time_ns()

        for size in subset_sizes:
            print(f"Processing subset size: {size}")
            subset_modalities = [modality[:size] for modality in modalities]
            subset_labels = truth_labels[:size]

            results = process_streaming_data(
                results,
                subset_modalities,
                window_size,
                k_neighbors,
                reduced_dim,
                n_clusters,
                seed,
                approach,
                subset_labels,
                step_window_ratio,
                noise_rate, 
                label_mode, 
                sorting
            )
            end_approach_time = time.time_ns()
            approach_processing_time = (end_approach_time - start_approach_time) / 1e9
            print(f"Processed with {approach} approach for {approach_processing_time} seconds")
        metrics[approach] = results

    print("Metrics:", metrics)

    output_generation.visualize_results(metrics, "subset_sizes", independent_variables, string_to_add=details_string)

    end_total_time = time.time_ns()

    total_processing_time = ((end_total_time - start_total_time) / 1e9 )/60

    print(f"Finished all processing for {details_string}")
    print(f"Total processing time: {total_processing_time} minutes")


if __name__ == "__main__":
    run()

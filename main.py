import time
import numpy as np
import data_loader
from matrix_operations import create_adjacency_matrix, fuse_matrices, perform_clustering, perform_svd_reduction
import metrics_evaluation
import output_generation
from collections import deque
from swfd import SeqBasedSWFD

def process_streaming_data(data_modalities, window_size, k_neighbors, reduced_dim, n_clusters, seed, approach, complete_true_labels, step_window_ratio=1):

    total_size = len(data_modalities[0])
    print(f"len(data_modalities[0]) = {len(data_modalities[0])}")
    window = deque(maxlen=window_size)
    
    sketches = []
    added_processing_time = 0

    results = metrics_evaluation.get_initial_results()

    # Setup SWFD first if that's the approach chosen
    if approach == "SWFD_first":
        start_time = time.process_time_ns()
        for modality in data_modalities:
            print(f"modality.shape: {modality.shape}")

            if modality.ndim == 1:
                modality = modality.reshape(-1, 1)

            # Determine R dynamically
            max_norm = np.max(np.linalg.norm(modality, axis=1)**2)
            sketches.append(SeqBasedSWFD(window_size, R=max_norm, d=modality.shape[1], sketch_dim=reduced_dim))
        end_time = time.process_time_ns()
        added_processing_time += (end_time - start_time)

    # Simulate streaming data, updating sketch with each arriving datapoint
    for i in range(total_size):
        # Collect a single data point from each modality
        data_point = [modality[i:i+1] for modality in data_modalities]
        window.append(data_point)

        if approach == "SWFD_first":
            start_time = time.process_time_ns()
            for j, sketch in enumerate(sketches):
                # print(f"data_point[{j}].shape: {data_point[j].shape}")
                sketch.fit(data_point[j])
            end_time = time.process_time_ns()
            added_processing_time += (end_time - start_time)

        # Only process once we have a full window
        if len(window) == window_size and (i + 1)*step_window_ratio % window_size == 0:
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

            if approach == "SWFD_first":
                # Get fused and not yet reduced matrix for norm delta comparison
                adjacency_matrices = []
                for m_index, modality in enumerate(data_modalities):
                    A_w = np.concatenate([point[m_index] for point in window], axis=0)
                    adjacency_matrices.append(create_adjacency_matrix(A_w, min(k_neighbors, A_w.shape[0]-1)))
                fused_matrix = fuse_matrices(adjacency_matrices)

            true_labels = complete_true_labels[i - len(reduced_matrix) + 1:i + 1]
            print(f"true_labels.shape: {true_labels.shape}, clusters.shape: {clusters.shape}")
            print(f"Unique labels: {np.unique(true_labels)}")

            results = metrics_evaluation.compute_all_metrics(results, i, fused_matrix, reduced_matrix, clusters, n_clusters, window_end_time, window_start_time, added_processing_time, true_labels)
            added_processing_time = 0
    return results

def run():
    # Parameters
    seed = 0
    subset_size = 10000
    window_size = 2000
    reduced_dim = 80
    k_neighbors = 50
    n_clusters = 0 #event related vs not #4  # technical, soccer, indignados
    step_window_ratio= 4
    # truth_mode = "binary"
    truth_mode = "types"
    # truth_mode = "all"
    sorted = False # True
    np.random.seed(seed)


    print("Loading dataset...")
    match truth_mode:
        case "binary":
            n_clusters = 2
            modalities, truth_labels = data_loader.load_sed2012_dataset(subset_size=subset_size, binary=True, event_types=True, sort_by_uploaded=sorted)
    
        case "types":
            n_clusters = 4
            modalities, truth_labels = data_loader.load_sed2012_dataset(subset_size=subset_size, binary=False, event_types=True, sort_by_uploaded=sorted)
    
        case _:
            n_clusters = 0
            modalities, truth_labels = data_loader.load_sed2012_dataset(subset_size=subset_size, binary=False, event_types=False, sort_by_uploaded=sorted)
    
    
    details_string = f"_mode={truth_mode},sorted={sorted},window={window_size},subset={subset_size},steps_per_window={step_window_ratio},clusters={n_clusters},k={k_neighbors},dim={reduced_dim}"

    approaches = [
        "naive", 
        "SVD", 
        # "SWFD_first"
        ]
    metrics = {}

    start_total_time = time.process_time_ns()

    for approach in approaches:
        print(f"Start processing with {approach} approach...")

        start_approach_time = time.process_time_ns()
        metrics[approach] = process_streaming_data(
            modalities,
            window_size,
            k_neighbors,
            reduced_dim,
            n_clusters,
            seed,
            approach,
            truth_labels,
            step_window_ratio
        )
        end_approach_time = time.process_time_ns()
        approach_processing_time = (end_approach_time - start_approach_time) / 1e9
        print(f"Processed with {approach} approach for {approach_processing_time} seconds")

    print("Metrics:", metrics)

    output_generation.visualize_results(metrics, string_to_add=details_string)

    output_generation.log_metrics(metrics, string_to_add=details_string)

    end_total_time = time.process_time_ns()

    total_processing_time = (end_total_time - start_total_time) / 1e9

    print(f"Finished all processing for {details_string}")
    print(f"Total processing time: {total_processing_time} seconds")


if __name__ == "__main__":
    run()

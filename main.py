import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import data_loader
from matrix_operations import create_adjacency_matrix, fuse_matrices, perform_clustering, incremental_dbscan_clustering, perform_hdbscan_clustering, perform_svd_reduction
import metrics_evaluation
import output_generation
from collections import deque
from swfd import SeqBasedSWFD

def process_streaming_data(results, data_modalities, modality_types, window_size, reduced_dim, n_clusters, seed, approach, complete_true_labels, step_window_ratio, noise_rate, label_mode, sorting):

    subset_size = len(data_modalities[0])

    total_start_time = time.time_ns()

    window = deque(maxlen=window_size)
    
    sketches = []

    # To store clusters and labels for the entire subset
    all_clusters = []
    all_true_labels = []

    previous_centroids = None
    previous_labels = None

    if approach == "SVD_incr":
        clustering_model = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, batch_size=window_size)

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

            true_labels = complete_true_labels[i - len(window) + 1:i + 1]
            all_true_labels.extend(true_labels)

            n_clusters = len(np.unique(all_true_labels))
            print(f"Amount of unique labels in this window: {n_clusters}")

            if approach == "SWFD_first":
                # First reduced via SWFD then fused
                adjacency_matrices = []
                for sketch in sketches:
                    B_t, _, _, _ = sketch.get()
                    A_w = np.concatenate([point[sketches.index(sketch)] for point in window])
                    adjacency_matrices.append(create_adjacency_matrix(B_t, modality_types[m_index]))
                reduced_matrix = fuse_matrices(adjacency_matrices)

            elif approach == "SWFD_after":
                # Fuse data
                adjacency_matrices = []
                for m_index, modality in enumerate(data_modalities):
                    A_w = np.concatenate([point[m_index] for point in window], axis=0)
                    adjacency_matrices.append(create_adjacency_matrix(A_w, modality_types[m_index]))
                fused_matrix = fuse_matrices(adjacency_matrices)
                print(f"fused matrix shape: {fused_matrix.shape}")

                # Reduce with SWFD sketching
                max_norm = np.max(np.linalg.norm(fused_matrix, axis=1)**2)
                swfd = SeqBasedSWFD(window_size, R=max_norm, d=fused_matrix.shape[1], sketch_dim=reduced_dim)
                for row in fused_matrix:
                    swfd.fit(row[np.newaxis, :])
                reduced_matrix, _, _, _ = swfd.get()
                if reduced_matrix.shape[0] != window_size:
                    print(f"reduced matrix shape: {reduced_matrix.shape}")
                    reduced_matrix = reduced_matrix.T
                    print(f"Tnsposed it to {reduced_matrix.shape}")

            else:
                # Fuse data
                adjacency_matrices = []
                for m_index, modality in enumerate(data_modalities):
                    A_w = np.concatenate([point[m_index] for point in window], axis=0)
                    adjacency_matrices.append(create_adjacency_matrix(A_w, modality_types[m_index]))
                fused_matrix = fuse_matrices(adjacency_matrices)

                # Reduce with SVD
                reduced_matrix = perform_svd_reduction(fused_matrix, reduced_dim, seed)

            # Clustering
            if approach == "SVD_incr":
                clusters = clustering_model.partial_fit(reduced_matrix).predict(reduced_matrix)
            elif approach == "DBSCAN":
                clusters, previous_centroids, previous_labels = incremental_dbscan_clustering(reduced_matrix, previous_centroids, previous_labels, eps=0.5, min_samples=5)
            else:
                clusters = perform_clustering(reduced_matrix, n_clusters, seed)

            # Accumulate all clusters
            all_clusters.extend(clusters)

    total_end_time = time.time_ns()

    # Compute metrics for the entire subset
    all_true_labels = np.array(all_true_labels)
    all_clusters = np.array(all_clusters)

    print(f"Amount of unique labels in total: {len(np.unique(all_true_labels))}")

    # Compute metrics for the entire subset
    results = metrics_evaluation.compute_all_metrics(results, subset_size, noise_rate, label_mode, sorting, all_clusters, all_true_labels, total_end_time, total_start_time)

    return results

def process_batch_data(results, data_modalities, modality_types, reduced_dim, n_clusters, seed, approach, complete_true_labels, noise_rate, label_mode, sorting):

    subset_size = len(data_modalities[0])

    total_start_time = time.time_ns()

    if approach == "SED":
        reduced_matrix = data_modalities
    else:
        # Fuse data
        adjacency_matrices = []
        for m_index, modality in enumerate(data_modalities):
            A_w = modality
            adjacency_matrices.append(create_adjacency_matrix(A_w, modality_types[m_index]))
        fused_matrix = fuse_matrices(adjacency_matrices)

        # Reduce with SVD
        reduced_matrix = perform_svd_reduction(fused_matrix, reduced_dim, seed)

    # Clustering
    if approach == "HDBSCAN_batch":
        all_clusters = perform_hdbscan_clustering(reduced_matrix)
    else:
        all_clusters = perform_clustering(reduced_matrix, n_clusters, seed)

    total_end_time = time.time_ns()

    # Compute metrics for the entire subset
    all_true_labels = np.array(complete_true_labels)
    all_clusters = np.array(all_clusters)

    print(f"Amount of unique labels in total: {len(np.unique(all_true_labels))}")

    # Compute metrics for the entire subset
    results = metrics_evaluation.compute_all_metrics(results, subset_size, noise_rate, label_mode, sorting, all_clusters, all_true_labels, total_end_time, total_start_time)

    return results

def run_experiment(experiment_type, variable_values, approaches, fixed_params, count):
    start_experiment_time = time.time_ns()
    params = fixed_params.copy()
    metrics = {}

    if experiment_type == "subset_size":
        # Only load the largest subset instead of doing it for every variable and approach
        all_modalities, modality_types, all_truth_labels = data_loader.load_sed2012_dataset(
                subset_size=max(variable_values), 
                binary=(params["label_mode"] == "binary"), 
                event_types=(params["label_mode"] != "all"), 
                sort_by_uploaded=params["sorting"], 
                noise_rate=params["noise_rate"]
            )

    for approach in approaches:
        results, independent_variables = metrics_evaluation.get_initial_results()
        start_approach_time = time.time_ns()

        for var_value in variable_values:
            print(f"Running experiment with {experiment_type} = {var_value} for {approach} approach")
            params[experiment_type] = var_value

            if experiment_type == "subset_size":
                modalities = [modality[:var_value] for modality in all_modalities]
                truth_labels = all_truth_labels[:var_value]
            else:
                modalities, modality_types, truth_labels = data_loader.load_sed2012_dataset(
                    subset_size=params["subset_size"], 
                    binary=(params["label_mode"] == "binary"), 
                    event_types=(params["label_mode"] != "all"), 
                    sort_by_uploaded=params["sorting"], 
                    noise_rate=params["noise_rate"]
                )

            params["noise_rate"] = np.sum(truth_labels == 0) / len(truth_labels)

            n_clusters = 2 if params["label_mode"] == "binary" else 4 if params["label_mode"] == "types" else 150

            if approach.endswith("_batch"):
                results = process_batch_data(
                    results=results,
                    data_modalities=modalities,
                    modality_types=modality_types,
                    reduced_dim=params["reduced_dim"],
                    n_clusters=n_clusters,
                    seed=params["seed"],
                    approach=approach,
                    complete_true_labels=truth_labels,
                    noise_rate=params["noise_rate"],
                    label_mode=params["label_mode"],
                    sorting=params["sorting"],
                )

            else:
                results = process_streaming_data(
                    results=results,
                    data_modalities=modalities,
                    modality_types=modality_types,
                    window_size=params["window_size"],
                    reduced_dim=params["reduced_dim"],
                    n_clusters=n_clusters,
                    seed=params["seed"],
                    approach=approach,
                    complete_true_labels=truth_labels,
                    step_window_ratio=params["step_window_ratio"],
                    noise_rate=params["noise_rate"],
                    label_mode=params["label_mode"],
                    sorting=params["sorting"],
                )
        
        end_approach_time = time.time_ns()
        approach_processing_time = (end_approach_time - start_approach_time) / 1e9
        print(f'Processed with {approach} approach, with {experiment_type}={var_value}  for {approach_processing_time} seconds')
        metrics[approach] = results
                
    details_string = f'mode={params["label_mode"]},sorted={params["sorting"]},noise={params["noise_rate"]},window={params["window_size"]},subset={params["subset_size"]},dim={params["reduced_dim"]}'
    output_generation.log_metrics(metrics=metrics, independent_variable=experiment_type, string_to_add=details_string, save_path= "logs/")
    output_generation.visualize_results(metrics=metrics, independent_variable=experiment_type, independent_variables=independent_variables, string_to_add=details_string, save_path="plots/")
    
    end_experiment_time = time.time_ns()
    experiment_processing_time = ((end_experiment_time - start_experiment_time) / 1e9 )/60

    print(f"Finished exp={experiment_type},{details_string} after {experiment_processing_time} minutes")

    return count + 1

if __name__ == "__main__":
    start_total_time = time.time_ns()
    seed = 0
    subset_sizes = [4000, 6000, 8000, 12000, 14000] #, 16000, 18000]
    noise_rates = [0.05, 0.25, 0.50, 0.75, .95] # [0.50, 0.75, .95] if higher base subset
    label_modes = ["binary", "types", "all"]
    sortings = [False, True]
    window_sizes = [500, 1000, 2000]
    count = 0

    np.random.seed(seed)
    fixed_params = {
        "seed": seed,
        "subset_size": subset_sizes[2],
        "noise_rate": noise_rates[2],
        "label_mode": label_modes[0],
        "sorting": sortings[0],
        "window_size": window_sizes[1],
        "reduced_dim": 80,
        "step_window_ratio": 1,
    }

    # Define experiments
    experiments = {
        "subset_size": subset_sizes,
        "noise_rate": noise_rates,
        "label_mode": label_modes,
        "sorting": sortings,
    }

    # Approaches
    approaches = [
        # "SWFD_after",
        # "DBSCAN_incr",
        # "HDBSCAN_batch",
        "SVD_incr",
        "SVD", 
        # "SVD_batch",
        # "SWFD_first",
        ]

    # Run experiments
    for experiment_type, variable_values in experiments.items():
        count = run_experiment(experiment_type, variable_values, approaches, fixed_params, count)

    end_total_time = time.time_ns()
    total_processing_time = ((end_total_time - start_total_time) / 1e9 )/60
    print(f"Finished running {count} experiments")
    print(f"Total processing time: {total_processing_time} minutes")
    print(f"Average per experiment: {total_processing_time/count} minutes")

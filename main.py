import time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import data_loader
from matrix_operations import create_adjacency_matrix, fuse_matrices, match_clusters, perform_dbscan_clustering, perform_svd_reduction, perform_clustering, perform_dbscan_incr_clustering, perform_hdbscan_clustering
import metrics_evaluation
import output_generation
from collections import deque
from swfd import SeqBasedSWFD
from incdbscan import IncrementalDBSCAN

def process_streaming_data(results, data_modalities, modality_types, window_size, reduced_dim, k_basis, n_clusters_total, seed, approach, complete_true_labels, step_window_ratio, noise_rate, label_mode, sorting, eps, min_samples):

    subset_size = len(data_modalities[0])

    total_start_time = time.time_ns()

    window = deque(maxlen=window_size)

    # To store clusters and labels for the entire subset
    all_clusters = []
    all_true_labels = []

    prev_centroids = None
    prev_clusters = None

    swfd = None
    clusterer = None

    # Simulate streaming data, updating sketch with each arriving datapoint
    for i in range(subset_size):
        # Collect a single data point from each modality
        data_point = [modality[i:i+1] for modality in data_modalities]
        window.append(data_point)

        # Only process once we have a full window
        if len(window) == window_size and (i + 1)*step_window_ratio % window_size == 0:
            print(f"i={i}")

            true_labels = complete_true_labels[i - len(window) + 1:i + 1]
            all_true_labels.extend(true_labels) #should be same as just taking complete_true_labels
            n_clusters = len(np.unique(true_labels))
            print(f"Amount of unique labels in this window: {n_clusters}")

            # Generate uni-modal adjacency matrices
            adjacency_matrices = []
            for m_index, _ in enumerate(data_modalities):
                A_w = np.concatenate([point[m_index] for point in window], axis=0)
                adjacency_matrices.append(create_adjacency_matrix(data=A_w, modality_type=modality_types[m_index], k_basis=k_basis))
            
            # Fuse data
            fused_matrix = fuse_matrices(adjacency_matrices)

            if approach == "SWFD":
                # Reduce with SWFD sketching
                if swfd is None: # Only gets run on first window
                    max_norm = np.max(np.linalg.norm(fused_matrix, axis=1)**2)
                    swfd = SeqBasedSWFD(N=window_size, R=max_norm, d=fused_matrix.shape[1], sketch_dim=reduced_dim)

                # Fit each of this window's adjacency matrix rows onto the swfd sketch
                for i in range(fused_matrix.shape[0]):
                    row = fused_matrix[i, :].reshape(1, -1)
                    swfd.fit(row)
                
                # Get the current sketch
                reduced_matrix, _, _, _ = swfd.get()

                # Has been returning it transposed, for some reason - transpose back if so
                if reduced_matrix.shape[0] != window_size:
                    print(f"reduced matrix shape: {reduced_matrix.shape}")
                    reduced_matrix = reduced_matrix.T
                    print(f"Transposed it to {reduced_matrix.shape}")
            else:
                # Reduce with SVD
                reduced_matrix = perform_svd_reduction(fused_matrix, reduced_dim, seed)

            # Clustering
            if approach == "SVD_incr":
                if clusterer is None:
                    clusterer = MiniBatchKMeans(n_clusters=n_clusters_total, random_state=seed, batch_size=window_size)
                clusters = clusterer.partial_fit(reduced_matrix).predict(reduced_matrix)

            elif approach == "DBSCAN_incr":
                if clusterer is None:
                    clusterer = IncrementalDBSCAN(eps=eps, min_pts=min_samples)
                # Insert batch of data points and get their labels
                clusters = clusterer.insert(reduced_matrix).get_cluster_labels(reduced_matrix)

            elif approach == "DBSCAN_centr":
                clusters, prev_centroids, prev_clusters = perform_dbscan_incr_clustering(reduced_matrix, prev_centroids, prev_clusters, eps=eps, min_samples=min_samples)
            
            else:
                clusters = perform_clustering(reduced_matrix, n_clusters, seed)

            print(f"clusters: {clusters}")

            if approach == "SVD_hung":
                clusters = match_clusters(prev_clusters, clusters, method="hungarian")
            
            if approach == "SVD_pot":
                clusters = match_clusters(prev_clusters, clusters, method="pot")

            prev_clusters = clusters
            # Accumulate all clusters
            all_clusters.extend(clusters)

    total_end_time = time.time_ns()

    # Compute metrics for the entire subset
    all_true_labels = np.array(all_true_labels)
    all_clusters = np.array(all_clusters)

    # Compute metrics for the entire subset
    results = metrics_evaluation.compute_all_metrics(results, subset_size, noise_rate, label_mode, sorting, reduced_dim, k_basis, window_size, all_clusters, all_true_labels, total_end_time, total_start_time)


    return results

def process_batch_data(results, data_modalities, modality_types, reduced_dim, k_basis, n_clusters, seed, approach, complete_true_labels, noise_rate, label_mode, sorting, eps, min_samples, min_cluster_size, window_size):

    subset_size = len(data_modalities[0])

    total_start_time = time.time_ns()

    # Fuse data
    adjacency_matrices = []
    for m_index, modality in enumerate(data_modalities):
        A_w = modality
        adjacency_matrices.append(create_adjacency_matrix(data=A_w, modality_type=modality_types[m_index], k_basis=k_basis))

    fused_matrix = fuse_matrices(adjacency_matrices)

    # Reduce with SVD
    reduced_matrix = perform_svd_reduction(fused_matrix, reduced_dim, seed)

    # Clustering
    if approach == "HDBSCAN_batch":
        all_clusters = perform_hdbscan_clustering(reduced_matrix, min_cluster_size=min_cluster_size, min_samples=min_samples)
    elif approach == "DBSCAN_batch":
        all_clusters = perform_dbscan_clustering(reduced_matrix, eps=eps, min_samples=min_samples)
    else:
        all_clusters = perform_clustering(reduced_matrix, n_clusters, seed)

    total_end_time = time.time_ns()

    # Compute metrics for the entire subset
    all_true_labels = np.array(complete_true_labels)
    all_clusters = np.array(all_clusters)

    print(f"Amount of unique labels in total: {len(np.unique(all_true_labels))}")

    # Compute metrics for the entire subset
    results = metrics_evaluation.compute_all_metrics(results, subset_size, noise_rate, label_mode, sorting, reduced_dim, k_basis, window_size, all_clusters, all_true_labels, total_end_time, total_start_time)

    return results

def run_experiment(df, experiment_type, variable_values, approaches, fixed_params, count):
    print(f"Running {experiment_type} experiment.")
    print(f"Fixed params: {fixed_params}")
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

            eps, min_samples, min_cluster_size = 1.5, 2, 3

            if approach.endswith("_batch"):
                results = process_batch_data(
                    results=results,
                    data_modalities=modalities,
                    modality_types=modality_types,
                    reduced_dim=params["reduced_dim"],
                    k_basis=params["k_basis"],
                    n_clusters=n_clusters,
                    seed=params["seed"],
                    approach=approach,
                    complete_true_labels=truth_labels,
                    noise_rate=params["noise_rate"],
                    label_mode=params["label_mode"],
                    sorting=params["sorting"],
                    eps=eps,
                    min_samples=min_samples,
                    min_cluster_size=min_cluster_size,
                    window_size=params["window_size"]
                )

            else:
                results = process_streaming_data(
                    results=results,
                    data_modalities=modalities,
                    modality_types=modality_types,
                    window_size=params["window_size"],
                    reduced_dim=params["reduced_dim"],
                    k_basis=params["k_basis"],
                    n_clusters_total=n_clusters,
                    seed=params["seed"],
                    approach=approach,
                    complete_true_labels=truth_labels,
                    step_window_ratio=params["step_window_ratio"],
                    noise_rate=params["noise_rate"],
                    label_mode=params["label_mode"],
                    sorting=params["sorting"],
                    eps=eps,
                    min_samples=min_samples
                )
        
        end_approach_time = time.time_ns()
        approach_processing_time = (end_approach_time - start_approach_time) / 1e9
        print(f'Processed with {approach} approach, with {experiment_type}={var_value}  for {approach_processing_time} seconds')
        metrics[approach] = results
                
    details_string = f'mode={params["label_mode"]},sorted={params["sorting"]},noise={params["noise_rate"]},window={params["window_size"]},subset={params["subset_size"]},dim={params["reduced_dim"]},k={params["k_basis"]}'
    output_generation.log_metrics(metrics=metrics, independent_variable=experiment_type, string_to_add=details_string, save_path= "logs/")
    output_generation.visualize_results(metrics=metrics, independent_variable=experiment_type, independent_variables=independent_variables, string_to_add=details_string, save_path="plots/")
    
    end_experiment_time = time.time_ns()
    experiment_processing_time = ((end_experiment_time - start_experiment_time) / 1e9 )/60

    print(f"Finished exp={experiment_type},{details_string} after {experiment_processing_time} minutes")

    return count + 1

if __name__ == "__main__":
    start_total_time = time.time_ns()
    seed = 0
    subset_sizes = [8000, 12000, 14000] #, 16000] # 18000]
    noise_rates = [0.05, 0.25, 0.50, 0.75, .95] # [0.50, 0.75, .95] if higher base subset
    label_modes = ["binary", "types", "all"]
    sortings = [False, True]
    window_sizes = [500, 1000, 2000, 4000]
    dims = [20, 50, 80, 100, 200]
    ks = [10, 20, 50, 80]
    count = 0

    np.random.seed(seed)
    fixed_params = {
        "seed": seed,
        "subset_size": subset_sizes[0],
        "noise_rate": noise_rates[-1],
        "label_mode": label_modes[0],
        "sorting": sortings[0],
        "window_size": window_sizes[1],
        "reduced_dim": 80,
        "step_window_ratio": 1,
    }

    # Define experiments
    experiments = {
        "demo": ["binary", "types"], #don't even bother with id-based lol
        "reduced_dim": dims,
        "k_basis": ks,
        "window_size": window_sizes,
        "label_mode": label_modes,
        "sorting": sortings,
        "noise_rate": noise_rates,
        "subset_size": subset_sizes,
        "noise_rate": noise_rates,
        "sorting": sortings,
    }

    # Approaches
    approaches = [
        "DBSCAN_incr",
        "DBSCAN_centr",
        "SWFD",
        "SVD_pot",
        "SVD_hung",
        "SVD_incr",
        "SVD", 
        "SVD_batch",
        "DBSCAN_batch",
        "HDBSCAN_batch",
        ]

    # Run experiments
    for experiment_type, variable_values in experiments.items():
        count = run_experiment(experiment_type, variable_values, approaches, fixed_params, count)

    end_total_time = time.time_ns()
    total_processing_time = ((end_total_time - start_total_time) / 1e9 )/60
    print(f"Finished running {count} experiments")
    print(f"Total processing time: {total_processing_time} minutes")
    print(f"Average per experiment: {total_processing_time/count} minutes")

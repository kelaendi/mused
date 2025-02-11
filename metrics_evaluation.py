from sklearn.metrics import (normalized_mutual_info_score, f1_score as sklearn_f1_score, precision_score, recall_score, accuracy_score, mean_absolute_error)

def get_initial_results():
    results = {
        # the actual metrics
        "f1_score": [],
        "nmi_score": [],
        "nmi_e_score": [],
        "precision": [],
        "recall": [],
        "accuracy": [],
        "mae": [],
        "processing_time": [],
        
        # possible independent variables
        "subset_size": [],
        "noise_rate":[],
        "label_mode":[],
        "sorting": [],
        "reduced_dim": [],
        "k_basis": [],
        "window_size": [],
    }

    independent_variables = [
        "subset_size",
        "noise_rate",
        "label_mode",
        "sorting",
        "reduced_dim",
        "k_basis",
        "window_size",
        ]
    return results, independent_variables

def compute_all_metrics(results, subset_size, noise_rate, label_mode, sorting, reduced_dim, k_basis, window_size, clusters, true_labels, end_time, start_time):
    log_string = ""
    
    results["subset_size"].append(subset_size)
    results["noise_rate"].append(noise_rate)
    results["label_mode"].append(label_mode)
    results["sorting"].append(sorting)
    results["reduced_dim"].append(reduced_dim)
    results["k_basis"].append(k_basis)
    results["window_size"].append(window_size)
    
    if "nmi_score" in results:
        nmi_score = normalized_mutual_info_score(true_labels, clusters)
        results["nmi_score"].append(nmi_score)
        log_string += f"nmi={nmi_score:.2f}, "

    if "nmi_e_score" in results:
        # Identify non-noise indices
        event_indices = [i for i, label in enumerate(true_labels) if label > 0]

        # Extract event-related true labels and predicted clusters
        true_labels_event = [true_labels[i] for i in event_indices]
        clusters_event = [clusters[i] for i in event_indices]

        # Compute NMI_e (NMI for only event-related instances)
        if len(set(true_labels_event)) > 1 and len(set(clusters_event)) > 1:  # Ensure there are at least two clusters
            nmi_e_score = normalized_mutual_info_score(true_labels_event, clusters_event)
        else:
            nmi_e_score = 0  # If there's only one event cluster, NMI is undefined

        results["nmi_e_score"].append(nmi_e_score)
        log_string += f"nmi_e={nmi_e_score:.2f}, "

    if "f1_score" in results:
        f1_score = sklearn_f1_score(true_labels, clusters, average='weighted', zero_division=0)
        results["f1_score"].append(f1_score)
        log_string += f"f1={f1_score:.2f}, "

    if "precision" in results:
        precision = precision_score(true_labels, clusters, average='weighted', zero_division=0)
        results["precision"].append(precision)
        log_string += f"precision={precision:.2f}, "

    if "recall" in results:
        recall = recall_score(true_labels, clusters, average='weighted', zero_division=0)
        results["recall"].append(recall)
        log_string += f"recall={recall:.2f}, "

    if "accuracy" in results:
        accuracy = accuracy_score(true_labels, clusters)
        results["accuracy"].append(accuracy)
        log_string += f"accuracy={accuracy:.2f}, "

    if "mae" in results:
        mae = mean_absolute_error(true_labels, clusters)
        results["mae"].append(mae)
        log_string += f"mae={mae:.2f}, "

    if "processing_time" in results:
        processing_time = (end_time - start_time) / 1e9
        results["processing_time"].append(processing_time)
        print(f"processing_time={processing_time}")
        log_string += f"processing_time={processing_time:.2f}"

    print(log_string)

    return results

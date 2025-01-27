import numpy as np
from sklearn.metrics import (normalized_mutual_info_score, f1_score as sklearn_f1_score, precision_score, recall_score, accuracy_score,
                             silhouette_score as sklearn_silhouette_score, davies_bouldin_score, average_precision_score, roc_auc_score, mean_absolute_error)

def get_initial_results():
    results = {
        "window_indices": [],
        "nmi_score": [],
        "f1_score": [],
        "precision": [],
        "recall": [],
        "accuracy": [],
        # "db_index": [],
        # "dunn_index": [],
        # "map_score": [],
        # "auc_score": [],
        "mae": [],
        # "silhouette_score": [],
        # "norm_delta": [],
        # "processing_time": [],
    }
    return results

def compute_all_metrics(results, i, fused_matrix, reduced_matrix, clusters, n_clusters, window_end_time, window_start_time, added_processing_time, true_labels):
    
    results["window_indices"].append(i)
    
    if "nmi_score" in results:
        nmi_score = normalized_mutual_info_score(true_labels, clusters)
        results["nmi_score"].append(nmi_score)
        print(f"nmi={nmi_score}")

    if "f1_score" in results:
        f1_score = sklearn_f1_score(true_labels, clusters, average='weighted', zero_division=0)
        results["f1_score"].append(f1_score)
        print(f"f1={f1_score}")

    if "precision" in results:
        precision = precision_score(true_labels, clusters, average='weighted', zero_division=0)
        results["precision"].append(precision)
        print(f"precision={precision}")

    if "recall" in results:
        recall = recall_score(true_labels, clusters, average='weighted', zero_division=0)
        results["recall"].append(recall)
        print(f"recall={recall}")

    if "accuracy" in results:
        accuracy = accuracy_score(true_labels, clusters)
        results["accuracy"].append(accuracy)
        print(f"accuracy={accuracy}")

    if "db_index" in results:
        db_index = davies_bouldin_score(reduced_matrix, clusters)
        results["db_index"].append(db_index)
        print(f"db_index={db_index}")

    if "dunn_index" in results:
        dunn_index = compute_dunn_index(reduced_matrix, clusters)
        results["dunn_index"].append(dunn_index)
        print(f"dunn_index={dunn_index}")


    if "map_score" in results or "auc_score" in results:
        # Generate prediction scores based on distances to cluster centers
        cluster_centers = np.array([reduced_matrix[clusters == c].mean(axis=0) for c in range(n_clusters)])
        predicted_scores = np.linalg.norm(reduced_matrix[:, None, :] - cluster_centers[clusters], axis=2).diagonal()

        if "map_score" in results:
            map_score = average_precision_score(true_labels, predicted_scores)
            results["map_score"].append(map_score)
            print(f"map={map_score}")

        if "auc_score" in results: 
            auc_score = roc_auc_score(true_labels, predicted_scores)
            results["auc_score"].append(auc_score)
            print(f"auc={auc_score}")

    if "mae" in results:
        mae = mean_absolute_error(true_labels, clusters)
        results["mae"].append(mae)
        print(f"mae={mae}")

    if "silhouette_score" in results:
        silhouette_score = sklearn_silhouette_score(reduced_matrix, clusters) if len(set(clusters)) > 1 else -1
        results["silhouette_score"].append(silhouette_score)
        print(f"silhouette={silhouette_score}")

    if "processing_time" in results:
        processing_time = (window_end_time - window_start_time + added_processing_time) / 1e9
        results["processing_time"].append(processing_time)
        print(f"processing_time={processing_time}")

    if "norm_delta" in results:
        fused_norm = np.linalg.norm(fused_matrix, 'fro')  # frobinius
        reduced_norm = np.linalg.norm(reduced_matrix, 'fro')
        norm_delta = abs(fused_norm - reduced_norm) / fused_norm
        results["norm_delta"].append(norm_delta)
        print(f"norm_delta={norm_delta}")

    return results

def compute_dunn_index(data, labels):
    unique_labels = np.unique(labels)
    centroids = [data[labels == c].mean(axis=0) for c in unique_labels]
    intra_cluster_dists = [np.mean(np.linalg.norm(
        data[labels == c] - centroids[i], axis=1)) for i, c in enumerate(unique_labels)]
    inter_cluster_dists = [np.linalg.norm(
        c1 - c2) for i, c1 in enumerate(centroids) for c2 in centroids[i+1:]]
    return min(inter_cluster_dists) / max(intra_cluster_dists)
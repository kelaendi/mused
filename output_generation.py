
import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD

def log_metrics(metrics, string_to_add="", save_path="logs/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log_file = os.path.join(save_path, "metric_averages"+string_to_add+".txt")

    approach_names = list(metrics.keys())
    header_line = "Metric & " + " & ".join(approach_names) + " \\\\\n"

    with open(log_file, "w") as f:
        f.write(header_line)
        # f.write("Metric & Naive & SVD & SWFD\_first \\\\\n")

        # Extract metric names from one approach
        metric_names = list(next(iter(metrics.values())).keys())
        metric_names.remove("window_indices")

        for metric_name in metric_names:
            metric_averages = []
            for approach, values in metrics.items():
                average = np.mean(values[metric_name])
                metric_averages.append(average)
            
            row = f"{metric_name.replace('_', ' ').capitalize()} & "
            row += " & ".join([f"{avg:.4f}" for avg in metric_averages])
            row += " \\\\\n"
            f.write(row)
    print(f"Metrics averages logged to {log_file}")

def visualize_results(metrics, string_to_add="", save_path="plots/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    metric_names = list(next(iter(metrics.values())).keys())
    
    for metric_name in metric_names:
        if metric_name == "window_indices":
            continue

        plt.figure(figsize=(10, 6))

        print(f"Plotting for metric: {metric_name}")
        
        for approach, values in metrics.items():
            print(f"Approach: {approach}, X: {values['window_indices']}, Y: {values[metric_name]}")
            if metric_name in values:
                plt.plot(values["window_indices"], values[metric_name], label=approach)
        metric_label = metric_name.replace('_', ' ').upper()
        plt.title(f"{metric_label} - Approach Comparison")
        plt.xlabel("Window End Index")
        plt.ylabel(metric_label)
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_path, f"{metric_name}_comparison{string_to_add}.png"))
        plt.close()

def visualize_clusters(reduced_matrix, clusters, plot_name="cluster_vis", save_path="plots/", string_to_add=""):
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

    plt.savefig(os.path.join(save_path, f"{plot_name}{string_to_add}.png"))
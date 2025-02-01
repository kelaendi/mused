import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD

def visualize_results(metrics, independent_variable, independent_variables, string_to_add="", save_path="plots/"):
    save_path += independent_variable
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    metric_names = [var for var in next(iter(metrics.values())).keys() if var not in independent_variables]
    
    for metric_name in metric_names:
        plt.figure(figsize=(10, 6))

        print(f"Plotting for metric: {metric_name}")
        
        for approach, values in metrics.items():
            print(f"Approach: {approach}, X: {values[independent_variable]}, Y: {values[metric_name]}")
            if metric_name in values:
                plt.plot(values[independent_variable], values[metric_name], label=approach)
        metric_label = metric_name.replace('_', ' ').upper()
        x_label = independent_variable.replace('_', ' ').upper()
        if metric_name == "processing_time":
            metric_label += " (s)"
        plt.title(f"{metric_label} BY {x_label} - APPROACH COMPARISON")
        plt.xlabel(x_label)
        plt.ylabel(metric_label)
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_path, f"{metric_name}_by_{independent_variable},{string_to_add}.png"))
        plt.close()

def log_averages(metrics, independent_variable="window_indices", string_to_add="", save_path="logs/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    log_file = os.path.join(save_path, f"metric_averages{string_to_add}.txt")

    approach_names = list(metrics.keys())
    header_line = "Metric Average & " + " & ".join(approach_names) + " \\\\\n"

    with open(log_file, "w") as f:
        f.write(header_line)

        metric_names = list(next(iter(metrics.values())).keys()).remove(independent_variable)

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

def log_metrics(metrics, independent_variable, string_to_add="", save_path="logs/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = f"exp={independent_variable},{string_to_add}"

    file = os.path.join(save_path, f"{filename}.txt")
    with open(file, "w") as f:
        f.write(f"{filename}\n\n")
        for approach, values in metrics.items():
            f.write(f"{approach}: {values}\n")

def generate_table(metrics, metric, independent_variable, string_to_add="", save_path="tables/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    table_file = os.path.join(save_path, f"{metric}_by_{independent_variable},{string_to_add}.txt")
    with open(table_file, "w") as f:
        f.write("\\begin{table}[h!]\n\\centering\n")
        f.write(f"\\caption{{{metric.replace('_', ' ').capitalize()} by {independent_variable.replace('_', ' ').capitalize()}}}\n")
        f.write("\\begin{tabular}{|l|" + "c|" * len(metrics.keys()) + "}\n\\hline\n")

        # Header row
        header = f"{independent_variable.replace('_', ' ').capitalize()} & " + " & ".join(metrics.keys()) + " \\\\\n"
        f.write(header)
        f.write("\\hline\n")

        # Data rows
        # Get unique values of the independent variable across all approaches
        unique_values = sorted(set(value for approach in metrics.values() for value in approach[independent_variable]))

        for unique_value in unique_values:
            row = [f"{unique_value}"]
            for approach, values in metrics.items():
                if unique_value in values[independent_variable]:
                    index = values[independent_variable].index(unique_value)
                    row.append(f"{values[metric][index]:.4f}")
                else:
                    row.append("N/A")
            f.write(" & ".join(row) + " \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table written to {table_file}")
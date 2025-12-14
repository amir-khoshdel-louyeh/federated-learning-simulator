import os
import ast
from typing import Dict, List, Any

import matplotlib.pyplot as plt


METRICS_KEYS = [
    "Accuracy",
    "Convergence",
    "Communication Cost",
    "Stability / Variance",
    "Training Time",
    "Velocity",
]


def read_results(result_file: str) -> Dict[str, Dict[str, List[Any]]]:
    results: Dict[str, Dict[str, List[Any]]] = {}
    if not os.path.exists(result_file):
        print(f"No result file found at {result_file}")
        return results
    with open(result_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            name, dict_str = line.split("=", 1)
            name = name.strip()
            try:
                data = ast.literal_eval(dict_str)
                # ensure keys exist
                for k in METRICS_KEYS:
                    data.setdefault(k, [])
                results[name] = data
            except Exception as e:
                print(f"Skipping line due to parse error: {e}\n{line}")
    return results


def compare_line_plots(results: Dict[str, Dict[str, List[Any]]]):
    # Create one figure per metric overlaying all algorithms
    for key in METRICS_KEYS:
        fig = plt.figure(figsize=(9, 5))
        plt.title(f"Comparison: {key}")
        for algo, metrics in results.items():
            vals = metrics.get(key, [])
            if not vals:
                continue
            plt.plot(range(1, len(vals) + 1), vals, marker="o", label=algo)
        plt.xlabel("Round")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        yield fig


def summary_bar_charts(results: Dict[str, Dict[str, List[Any]]]):
    # For each metric, show a bar chart summarizing the final value per algorithm
    for key in METRICS_KEYS:
        labels = []
        values = []
        for algo, metrics in results.items():
            vals = metrics.get(key, [])
            if not vals:
                continue
            labels.append(algo)
            values.append(vals[-1])
        if not values:
            continue
        fig = plt.figure(figsize=(9, 5))
        plt.title(f"Final {key} per Algorithm")
        plt.bar(labels, values)
        plt.xticks(rotation=30, ha="right")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        yield fig


def convergence_speed_plot(results: Dict[str, Dict[str, List[Any]]], target_accuracy: float = None):
    # Show rounds needed to reach a target accuracy; default = best final accuracy among algorithms
    final_acc = [metrics.get("Accuracy", [])[-1] for metrics in results.values() if metrics.get("Accuracy", [])]
    if not final_acc:
        return
    if target_accuracy is None:
        target_accuracy = max(final_acc)
    labels = []
    rounds_to_target = []
    for algo, metrics in results.items():
        acc_vals = metrics.get("Accuracy", [])
        if not acc_vals:
            continue
        labels.append(algo)
        r = next((i + 1 for i, a in enumerate(acc_vals) if a >= target_accuracy), None)
        rounds_to_target.append(r if r is not None else float('inf'))
    if not labels:
        return
    fig = plt.figure(figsize=(9, 5))
    plt.title(f"Rounds to reach target accuracy = {target_accuracy:.4f}")
    plt.bar(labels, rounds_to_target)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Rounds")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def efficiency_plot(results: Dict[str, Dict[str, List[Any]]]):
    # Scatter plot: Final Accuracy vs Final Communication Cost (efficiency frontier)
    labels = []
    acc = []
    comm = []
    for algo, metrics in results.items():
        a_vals = metrics.get("Accuracy", [])
        c_vals = metrics.get("Communication Cost", [])
        if not a_vals or not c_vals:
            continue
        labels.append(algo)
        acc.append(a_vals[-1])
        comm.append(c_vals[-1])
    if not labels:
        return
    fig = plt.figure(figsize=(9, 5))
    plt.title("Efficiency: Final Accuracy vs Final Communication Cost")
    plt.scatter(comm, acc)
    for i, lbl in enumerate(labels):
        plt.annotate(lbl, (comm[i], acc[i]), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Final Communication Cost (bytes, cumulative)")
    plt.ylabel("Final Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def stability_plot(results: Dict[str, Dict[str, List[Any]]]):
    # Compare stability by averaging variance across rounds
    labels = []
    avg_var = []
    for algo, metrics in results.items():
        v_vals = metrics.get("Stability / Variance", [])
        if not v_vals:
            continue
        labels.append(algo)
        avg_var.append(sum(v_vals) / len(v_vals))
    if not labels:
        return
    fig = plt.figure(figsize=(9, 5))
    plt.title("Average Stability/Variance across rounds")
    plt.bar(labels, avg_var)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Average Variance (norm-based)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


def generate_figures(results: Dict[str, Dict[str, List[Any]]]):
    """Generate all comparison figures without displaying them.
    Returns a list of matplotlib Figure objects.
    """
    figs = []
    # line comparisons
    for fig in compare_line_plots(results):
        figs.append(fig)
    # final value summaries
    for fig in summary_bar_charts(results):
        figs.append(fig)
    # convergence speed
    fig = convergence_speed_plot(results)
    if fig is not None:
        figs.append(fig)
    # efficiency frontier
    fig = efficiency_plot(results)
    if fig is not None:
        figs.append(fig)
    # stability summary
    fig = stability_plot(results)
    if fig is not None:
        figs.append(fig)
    return figs


def main():
    here = os.path.dirname(__file__)
    result_file = os.path.join(here, "result.txt")
    results = read_results(result_file)
    if not results:
        print("No results to compare. Run training first to generate result.txt.")
        return

    # Core comparisons (standalone mode)
    figs = generate_figures(results)
    print(f"Displaying {len(figs)} comparison graphs... Close figures to exit.")
    for fig in figs:
        fig.show()
    plt.show()


if __name__ == "__main__":
    main()

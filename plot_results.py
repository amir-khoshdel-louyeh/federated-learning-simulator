import os
import ast
import matplotlib.pyplot as plt


def read_results(result_file: str):
    results = {}
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
                results[name] = data
            except Exception as e:
                print(f"Skipping line due to parse error: {e}\n{line}")
    return results


def plot_individual(results):
    # For each algorithm, plot its metrics in a multi-subplot figure
    for algo, metrics in results.items():
        fig, axes = plt.subplots(3, 2, figsize=(10, 8))
        fig.suptitle(f"Algorithm: {algo}")
        keys = [
            "Accuracy",
            "Convergence",
            "Communication Cost",
            "Stability / Variance",
            "Training Time",
            "Velocity",
        ]
        for idx, key in enumerate(keys):
            ax = axes[idx // 2][idx % 2]
            vals = metrics.get(key, [])
            ax.plot(range(1, len(vals) + 1), vals, marker="o")
            ax.set_title(key)
            ax.set_xlabel("Round")
            ax.grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def plot_comparisons(results):
    # Create one figure per metric, overlaying all algorithms
    keys = [
        "Accuracy",
        "Convergence",
        "Communication Cost",
        "Stability / Variance",
        "Training Time",
        "Velocity",
    ]
    for key in keys:
        plt.figure(figsize=(8, 5))
        plt.title(f"Comparison: {key}")
        for algo, metrics in results.items():
            vals = metrics.get(key, [])
            if not vals:
                continue
            plt.plot(range(1, len(vals) + 1), vals, marker="o", label=algo)
        plt.xlabel("Round")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()


def main():
    here = os.path.dirname(__file__)
    result_file = os.path.join(here, "result.txt")
    results = read_results(result_file)
    if not results:
        print("No results to plot.")
        return

    # per-algorithm plots
    plot_individual(results)
    # comparison plots per metric
    plot_comparisons(results)

    print("Displaying plots... Close figures to exit.")
    plt.show()


if __name__ == "__main__":
    main()

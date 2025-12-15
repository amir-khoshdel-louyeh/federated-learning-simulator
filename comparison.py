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



def generate_figures(results: Dict[str, Dict[str, List[Any]]]):
    """Generate all comparison figures without displaying them.
    Returns a list of matplotlib Figure objects.
    """
    figs = []
    # Only line comparisons per metric across rounds
    for fig in compare_line_plots(results):
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

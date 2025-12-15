import os
import ast
from typing import Dict, List, Any, Optional, Tuple

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


def _tag_results(results: Dict[str, Dict[str, List[Any]]], tag: str) -> Dict[str, Dict[str, List[Any]]]:
    # Return a shallow-cloned dict with algo names tagged, e.g., 'fedavg (Py)'
    tagged: Dict[str, Dict[str, List[Any]]] = {}
    for algo, metrics in results.items():
        tagged[f"{algo} ({tag})"] = metrics
    return tagged


def generate_figures_from_paths(paths: Optional[List[str]] = None) -> List[Any]:
    """Generate figures depending on availability of p_result.txt and m_result.txt (or provided paths).

    Behavior:
    - If only one file has results: produce one set of figures (one per metric).
    - If both exist: for each metric, produce three figures in order:
        1) Python-only (p_result)
        2) Matlab-only (m_result)
        3) Combined overlay (p_result tagged '(Py)' and m_result tagged '(Mat)')
    Returns list of matplotlib Figure objects in the order above for each metric.
    """
    here = os.path.dirname(__file__)
    # Default search if no paths provided (prefer ../results/ directory)
    if not paths:
        results_dir = os.path.normpath(os.path.join(here, "..", "results"))
        candidates = [
            os.path.join(results_dir, "p_result.txt"),
            os.path.join(results_dir, "m_result.txt"),
            os.path.join(results_dir, "result.txt"),  # legacy fallback in results/
            os.path.join(here, "result.txt"),         # legacy fallback in this folder
        ]
        # Deduplicate while preserving order
        seen = set()
        paths = []
        for p in candidates:
            if p not in seen and os.path.exists(p):
                paths.append(p)
                seen.add(p)

    # Identify specific known files if present
    p_path = None
    m_path = None
    legacy_only = []
    for p in paths:
        base = os.path.basename(p)
        if base == "p_result.txt":
            p_path = p
        elif base == "m_result.txt":
            m_path = p
        elif base == "result.txt":
            legacy_only.append(p)

    figs: List[Any] = []

    if p_path and m_path:
        py = read_results(p_path)
        ma = read_results(m_path)
        # 1) Python-only
        for key in METRICS_KEYS:
            fig = plt.figure(figsize=(9, 5))
            plt.title(f"Comparison: {key} — Python (p_result)")
            for algo, metrics in py.items():
                vals = metrics.get(key, [])
                if not vals:
                    continue
                plt.plot(range(1, len(vals) + 1), vals, marker="o", label=algo)
            plt.xlabel("Round")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            figs.append(fig)
        # 2) Matlab-only
        for key in METRICS_KEYS:
            fig = plt.figure(figsize=(9, 5))
            plt.title(f"Comparison: {key} — Matlab (m_result)")
            for algo, metrics in ma.items():
                vals = metrics.get(key, [])
                if not vals:
                    continue
                plt.plot(range(1, len(vals) + 1), vals, marker="o", label=algo)
            plt.xlabel("Round")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            figs.append(fig)
        # 3) Combined overlay
        both = {}
        both.update(_tag_results(py, "Py"))
        both.update(_tag_results(ma, "Mat"))
        for key in METRICS_KEYS:
            fig = plt.figure(figsize=(9, 5))
            plt.title(f"Comparison: {key} — Both (p_result vs m_result)")
            for algo, metrics in both.items():
                vals = metrics.get(key, [])
                if not vals:
                    continue
                plt.plot(range(1, len(vals) + 1), vals, marker="o", label=algo)
            plt.xlabel("Round")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            figs.append(fig)
        return figs

    # Only one results file present (or legacy)
    chosen_path = p_path or m_path or (legacy_only[0] if legacy_only else None)
    if not chosen_path:
        print("No results to compare. Run training first to generate m_result.txt or p_result.txt.")
        return figs

    results = read_results(chosen_path)
    for fig in compare_line_plots(results):
        figs.append(fig)
    return figs


def main():
    here = os.path.dirname(__file__)
    # Prefer p_result and m_result under ../results; fallback to legacy result.txt
    results_dir = os.path.normpath(os.path.join(here, "..", "results"))
    candidates = [
        os.path.join(results_dir, "p_result.txt"),
        os.path.join(results_dir, "m_result.txt"),
        os.path.join(results_dir, "result.txt"),
        os.path.join(here, "result.txt"),
    ]
    figs = generate_figures_from_paths([p for p in candidates if os.path.exists(p)])
    if not figs:
        print("No results to compare. Run training first to generate m_result.txt or p_result.txt.")
        return
    print(f"Displaying {len(figs)} comparison graphs... Close figures to exit.")
    for fig in figs:
        fig.show()
    plt.show()


if __name__ == "__main__":
    main()

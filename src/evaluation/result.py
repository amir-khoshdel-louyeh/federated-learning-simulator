def _prepare_results(results):
    processed = {}
    for key, value in results.items():
        if isinstance(value, list) and value:
            processed[key] = sum(value) / len(value)
        else:
            processed[key] = value
    return sorted(processed.items(), key=lambda item: item[1] if item[1] is not None else float('-inf'), reverse=True)


def format_sorted_results(results):
    """Return a formatted leaderboard string sorted from best to worst."""
    sorted_items = _prepare_results(results)
    lines = ["===== Algorithm Results (Best -> Worst) ====="]
    for rank, (algo, acc) in enumerate(sorted_items, 1):
        if acc is None:
            lines.append(f"{rank}. {algo}: n/a")
        else:
            lines.append(f"{rank}. {algo}: {acc:.4f}")
    lines.append("==========================================")
    return "\n".join(lines)


def print_sorted_results(results):
    print("\n" + format_sorted_results(results) + "\n")

# Example usage:
if __name__ == "__main__":
    # Example
    results = {
        'FedAvg': 0.73,
        'FedOpt': 0.74,
        'FedProx': 0.72,
        'FedNova': 0.71
    }
    print_sorted_results(results)

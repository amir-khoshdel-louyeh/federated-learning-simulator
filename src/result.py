def print_sorted_results(results):
    """
    results: dict mapping algorithm name to accuracy (float or list of floats)
    """
    # If value is a list, take mean (for FedPer)
    processed = {}
    for k, v in results.items():
        if isinstance(v, list):
            processed[k] = sum(v) / len(v)
        else:
            processed[k] = v
    # Sort by accuracy descending
    sorted_items = sorted(processed.items(), key=lambda x: x[1], reverse=True)
    print("\n===== Sorted Algorithm Results =====")
    for i, (algo, acc) in enumerate(sorted_items, 1):
        print(f"{i}. {algo}: {acc:.4f}")
    print("===================================\n")

# Example usage:
if __name__ == "__main__":
    # Example
    results = {
        'FedAvg': 0.73,
        'FedOpt': 0.74,
        'FedPer': [0.41, 0.56, 0.40],
        'FedProx': 0.72,
        'FedSGD': 0.11
    }
    print_sorted_results(results)

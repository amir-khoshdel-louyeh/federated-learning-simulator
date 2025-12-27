import numpy as np
from typing import List


def make_label_shards(y_train: np.ndarray, clients: int, classes_per_client: int = 2, seed: int | None = None) -> List[np.ndarray]:
    """Create non-IID shards by assigning labels to clients and splitting indices.

    - Each client is assigned `classes_per_client` labels.
    - For each label, its indices are split among the clients that selected it.
    - Ensures disjoint assignment (each sample belongs to exactly one client).

    Returns a list of numpy index arrays, one per client.
    """
    rng = np.random.default_rng(seed)
    labels = np.unique(y_train)
    L = len(labels)
    # Clients choose labels (with coverage)
    chosen = [set() for _ in range(clients)]
    # Round-robin to guarantee coverage, then random fill
    li = 0
    for c in range(clients):
        while len(chosen[c]) < classes_per_client and li < L:
            chosen[c].add(labels[li])
            li += 1
    # Fill remaining choices randomly
    for c in range(clients):
        while len(chosen[c]) < classes_per_client:
            chosen[c].add(rng.choice(labels))
    # Build reverse mapping: label -> clients that selected it
    label_to_clients: dict[int, List[int]] = {int(lbl): [] for lbl in labels}
    for c, labs in enumerate(chosen):
        for lbl in labs:
            label_to_clients[int(lbl)].append(c)
    # If any label unassigned (shouldn't happen), assign to random client
    for lbl in labels:
        if len(label_to_clients[int(lbl)]) == 0:
            label_to_clients[int(lbl)] = [rng.integers(0, clients)]
    # Prepare shards
    shards: List[List[int]] = [[] for _ in range(clients)]
    # For each label, split its indices across its selected clients
    for lbl in labels:
        idx = np.where(y_train == lbl)[0]
        rng.shuffle(idx)
        assigned_clients = label_to_clients[int(lbl)]
        parts = np.array_split(idx, len(assigned_clients))
        for cli, part in zip(assigned_clients, parts):
            shards[cli].extend(part.tolist())
    # Convert to numpy arrays and shuffle per client
    shards_np: List[np.ndarray] = []
    for s in shards:
        arr = np.array(s, dtype=int)
        rng.shuffle(arr)
        shards_np.append(arr)
    return shards_np

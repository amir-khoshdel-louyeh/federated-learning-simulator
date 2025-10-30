import numpy as np

def split_indices_iid(num_samples, num_clients):
    indices = np.random.permutation(num_samples)
    return np.array_split(indices, num_clients)

def split_indices_non_iid(num_samples, num_clients, num_shards=2, num_classes=10, labels=None):
    # Non-IID partitioning: each client gets data from a few classes only
    assert labels is not None, "Labels must be provided for non-IID partitioning."
    shards_per_client = num_shards
    shard_size = num_samples // (num_clients * shards_per_client)
    idxs = np.arange(num_samples)
    labels = np.array(labels)
    # Sort indices by label
    idxs_labels = np.vstack((idxs, labels)).T
    idxs_labels = idxs_labels[idxs_labels[:,1].argsort()]
    idxs = idxs_labels[:,0]
    # Divide into shards
    shards = [idxs[i*shard_size:(i+1)*shard_size] for i in range((num_clients * shards_per_client))]
    client_indices = []
    for i in range(num_clients):
        shard_idxs = np.random.choice(len(shards), shards_per_client, replace=False)
        client_shards = [shards[j] for j in shard_idxs]
        client_indices.append(np.concatenate(client_shards))
        # Remove assigned shards
        for j in sorted(shard_idxs, reverse=True):
            del shards[j]
    return client_indices

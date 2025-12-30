import time
import numpy as np
from model import make_model
from metrics import compute_round_metrics


def federated_average(x_train, y_train, x_test, y_test, clients=3, rounds=3, report=None, shards=None):
    len_xtrain = len(x_train)
    if shards is None:
        idx = np.random.permutation(len_xtrain)
        shards = np.array_split(idx, clients)

    global_model = make_model()
    global_weights = global_model.get_weights()
    # approximate per-round communication cost (upload+download of weights)
    comm_cost = 0
    prev_acc = None
    prev_global_weights = [np.copy(w) for w in global_weights]

    for i in range(rounds):
        round_start = time.perf_counter()
        client_weights = []
        for shard in shards:
            local = make_model()
            local.set_weights(global_weights)
            local.fit(x_train[shard], y_train[shard], epochs=1, batch_size=64, verbose=0)
            client_weights.append(local.get_weights())

        new_weights = []
        for weights in zip(*client_weights):
            new_weights.append(np.mean(weights, axis=0))
        global_weights = new_weights

        metrics, prev_acc, prev_global_weights, comm_cost = compute_round_metrics(
            "FedAvg",
            i + 1,
            round_start,
            global_model,
            x_test,
            y_test,
            prev_acc,
            prev_global_weights,
            global_weights,
            clients,
            comm_cost,
            local_weights_list=client_weights,
            reference_weights=global_weights,
        )
        if callable(report):
            report("FedAvg", i + 1, metrics)

    global_model.set_weights(global_weights)
    loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
    return float(acc)

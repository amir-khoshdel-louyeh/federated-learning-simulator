import numpy as np
from model import make_model


def federated_average(x_train, y_train, x_test, y_test, clients=3, rounds=3, full_data_per_client=False):
    len_xtrain = len(x_train)
    if full_data_per_client:
        shards = []
        for client_id in range(clients):
            shards.append(np.arange(len_xtrain))
    else:
        idx = np.random.permutation(len_xtrain)
        shards = np.array_split(idx, clients)

    global_model = make_model()
    global_weights = global_model.get_weights()

    for i in range(rounds):
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

    global_model.set_weights(global_weights)
    loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
    return float(acc)

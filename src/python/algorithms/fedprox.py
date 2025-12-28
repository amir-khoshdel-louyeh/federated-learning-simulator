import time
import numpy as np
import tensorflow as tf

from model import make_model
from metrics import compute_round_metrics


def fedprox(x_train, y_train, x_test, y_test, clients=3, rounds=3, mu=0.01, local_epochs=1, batch_size=64, full_data_per_client=False, report=None, shards=None):
    """Simple FedProx implementation.

    - Splits `x_train` among `clients` (unless `full_data_per_client=True`).
    - For each round, each client initializes from the global model and
      performs `local_epochs` of gradient updates with an additional
      proximal term (mu/2 * ||w - w_global||^2) added to the loss.
    - After local updates, client weights are averaged (FedAvg-style) to
      form the new global weights.

    Returns test accuracy (float) on (x_test, y_test).
    """
    len_xtrain = len(x_train)
    if shards is None:
        if full_data_per_client:
            shards = [np.arange(len_xtrain) for _ in range(clients)]
        else:
            idx = np.random.permutation(len_xtrain)
            shards = np.array_split(idx, clients)

    # initialize global model and weights
    global_model = make_model()
    global_weights = global_model.get_weights()
    comm_cost = 0
    prev_acc = None
    prev_global_weights = [np.copy(w) for w in global_weights]

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for r in range(rounds):
        round_start = time.perf_counter()
        client_weights = []

        # convert global weights to tensors for proximal term
        global_tensors = [tf.convert_to_tensor(w) for w in global_weights]

        for shard in shards:
            # skip empty shards
            if len(shard) == 0:
                continue

            local = make_model()
            local.set_weights(global_weights)

            optimizer = tf.keras.optimizers.Adam()

            # get trainable variables (these correspond in order to weight arrays for our simple model)
            train_vars = local.trainable_variables

            # local training loop
            for epoch in range(local_epochs):
                # shuffle shard indices for local epoch
                perm = np.random.permutation(shard)
                for start in range(0, len(perm), batch_size):
                    batch_idx = perm[start:start + batch_size]
                    x_batch = tf.convert_to_tensor(x_train[batch_idx])
                    y_batch = tf.convert_to_tensor(y_train[batch_idx])

                    with tf.GradientTape() as tape:
                        preds = local(x_batch, training=True)
                        base_loss = loss_fn(y_batch, preds)

                        # proximal term: mu/2 * sum ||w - w_global||^2 over trainable weights
                        prox = 0.0
                        # build mapping between train_vars and global_tensors by shapes/order
                        # NOTE: this assumes the trainable variables are in the same order as
                        # the corresponding entries in local.get_weights(); for our model it holds.
                        local_weights = local.get_weights()
                        # iterate only over trainable variables count
                        t = 0
                        for var in train_vars:
                            gw = tf.convert_to_tensor(local_weights[t])  # corresponding global initial weight
                            prox += tf.reduce_sum(tf.square(var - gw))
                            t += 1

                        prox_term = (mu / 2.0) * prox
                        loss = base_loss + prox_term

                    grads = tape.gradient(loss, train_vars)
                    # apply grads
                    optimizer.apply_gradients(zip(grads, train_vars))

            client_weights.append(local.get_weights())

        # average client weights
        new_weights = []
        for weights in zip(*client_weights):
            new_weights.append(np.mean(weights, axis=0))
        global_weights = new_weights

        metrics, prev_acc, prev_global_weights, comm_cost = compute_round_metrics(
            "FedProx",
            r + 1,
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
            report("FedProx", r + 1, metrics)

    # evaluate
    global_model.set_weights(global_weights)
    loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
    return float(acc)

import time
import numpy as np
import tensorflow as tf

from model import make_model
from metrics import compute_round_metrics


def scaffold(x_train, y_train, x_test, y_test, clients=3, rounds=3, local_epochs=1, batch_size=64, lr=0.001, full_data_per_client=False, report=None):
    """A straightforward SCAFFOLD implementation.

    Notes / simplifications:
    - Uses per-variable control variates stored as lists of numpy arrays (same shapes as model.get_weights()).
    - Local updates use GradientTape and apply the correction (server_c - client_c) to gradients.
    - Client control variates are updated following the SCAFFOLD update rule
      c_i^{t+1} = c_i^t - c^t + (1 / (steps * lr)) * (w^t - w_i^{t+1})
      where steps is the number of local gradient steps taken.
    - Server control variate is updated by averaging client changes.

    Returns test accuracy (float).
    """
    n = len(x_train)
    if full_data_per_client:
        shards = [np.arange(n) for _ in range(clients)]
    else:
        idx = np.random.permutation(n)
        shards = np.array_split(idx, clients)

    # initialize global model and weights
    global_model = make_model()
    global_weights = global_model.get_weights()
    comm_cost = 0
    prev_acc = None
    prev_global_weights = [np.copy(w) for w in global_weights]

    # initialize server control variate c and per-client c_i as zero arrays matching weights
    c = [np.zeros_like(w) for w in global_weights]
    c_i_list = [[np.zeros_like(w) for w in global_weights] for _ in range(clients)]

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for rnd in range(rounds):
        round_start = time.perf_counter()
        client_weights = []
        new_c_i_list = []

        for client_idx, shard in enumerate(shards):
            if len(shard) == 0:
                # keep client control variate unchanged and skip
                new_c_i_list.append(c_i_list[client_idx])
                continue

            local = make_model()
            local.set_weights(global_weights)

            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

            train_vars = local.trainable_variables

            # number of local gradient steps (for control variate update)
            steps = 0

            for epoch in range(local_epochs):
                perm = np.random.permutation(shard)
                for start in range(0, len(perm), batch_size):
                    batch_idx = perm[start:start + batch_size]
                    x_batch = tf.convert_to_tensor(x_train[batch_idx])
                    y_batch = tf.convert_to_tensor(y_train[batch_idx])

                    with tf.GradientTape() as tape:
                        preds = local(x_batch, training=True)
                        base_loss = loss_fn(y_batch, preds)

                    grads = tape.gradient(base_loss, train_vars)

                    # Build correction list aligned with train_vars order.
                    # We map train_vars to local.get_weights() entries by index order.
                    local_weights = local.get_weights()
                    # prepare server and client control tensors for train_vars
                    corr_tensors = []
                    t = 0
                    for var in train_vars:
                        # use corresponding weight array (assumes ordering matches)
                        server_c = tf.convert_to_tensor(c[t])
                        client_c = tf.convert_to_tensor(c_i_list[client_idx][t])
                        corr = server_c - client_c
                        corr_tensors.append(corr)
                        t += 1

                    # apply corrected gradients: grad + corr
                    corrected_grads = [g + corr for g, corr in zip(grads, corr_tensors)]
                    optimizer.apply_gradients(zip(corrected_grads, train_vars))
                    steps += 1

            # save local weights
            w_local = local.get_weights()
            client_weights.append(w_local)

            # update client control variate c_i
            old_c_i = c_i_list[client_idx]
            new_c_i = []
            # formula: c_i' = c_i - c + (1/(steps*lr)) * (w_global - w_local)
            scale = 1.0 / (max(1, steps) * lr)
            for gw, wloc, ci in zip(global_weights, w_local, old_c_i):
                delta = gw - wloc
                new_ci = ci - gw*0  # placeholder to copy shape; we'll compute below
                # compute numeric arrays
                new_ci = ci - np.array(c[0])  # temporary, will be replaced per-entry
                break

            # The above per-entry loop is easier to implement explicitly:
            new_c_i = []
            for gw, wloc, ci, gc in zip(global_weights, w_local, old_c_i, c):
                new_ci = ci - gc + scale * (gw - wloc)
                new_c_i.append(new_ci)

            new_c_i_list.append(new_c_i)

        # average client weights (standard FedAvg) to form new global weights
        # handle possible different client counts if some shards empty
        if len(client_weights) == 0:
            break
        new_weights = []
        for weights in zip(*client_weights):
            new_weights.append(np.mean(weights, axis=0))
        # update server control variate: c = c + (1/K) * sum_i (c_i' - c_i)
        sum_delta = [np.zeros_like(w) for w in global_weights]
        for old_ci, new_ci in zip(c_i_list, new_c_i_list):
            for j, (oci, nci) in enumerate(zip(old_ci, new_ci)):
                sum_delta[j] += (nci - oci)
        K = len(new_c_i_list)
        c = [ci + (sum_delta_j / K) for ci, sum_delta_j in zip(c, sum_delta)]

        # set for next round
        global_weights = new_weights
        c_i_list = new_c_i_list

        metrics, prev_acc, prev_global_weights, comm_cost = compute_round_metrics(
            "SCAFFOLD",
            rnd + 1,
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
            report("SCAFFOLD", rnd + 1, metrics)

    # final evaluation
    global_model.set_weights(global_weights)
    loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
    return float(acc)

import time
import numpy as np
from model import make_model
from metrics import compute_round_metrics


def fednova(x_train, y_train, x_test, y_test, clients=3, rounds=3, local_epochs=1,
            batch_size=64, full_data_per_client=False, report=None):
    """FedNova: Normalized Averaging for Heterogeneous Client Updates.
    
    Addresses the objective inconsistency problem in federated learning when
    clients perform different numbers of local steps. Uses normalized averaging
    where each client's update is weighted by the effective number of steps
    (tau_i) they took relative to a normalization factor.
    
    The aggregation rule:
    w_{t+1} = w_t - (1/sum_i(tau_i)) * sum_i(tau_i * (w_t - w_i))
    
    where tau_i is the number of local steps for client i.
    
    Returns:
        Test accuracy (float)
    """
    n = len(x_train)
    if full_data_per_client:
        shards = [np.arange(n) for _ in range(clients)]
    else:
        idx = np.random.permutation(n)
        shards = np.array_split(idx, clients)
    
    # Initialize global model
    global_model = make_model()
    global_weights = global_model.get_weights()
    comm_cost = 0
    prev_acc = None
    prev_global_weights = [np.copy(w) for w in global_weights]
    
    for rnd in range(rounds):
        round_start = time.perf_counter()
        client_deltas = []  # (delta_weights, tau_i) for each client
        
        for shard in shards:
            if len(shard) == 0:
                continue
            
            local = make_model()
            local.set_weights(global_weights)
            
            # Count actual training steps (batches processed)
            num_batches = int(np.ceil(len(shard) / batch_size))
            tau_i = num_batches * local_epochs  # effective number of local steps
            
            local.fit(x_train[shard], y_train[shard], epochs=local_epochs,
                     batch_size=batch_size, verbose=0)
            
            local_weights = local.get_weights()
            
            # Compute delta: w_t - w_i
            delta = [w_global - w_local for w_global, w_local in zip(global_weights, local_weights)]
            
            client_deltas.append((delta, tau_i))
        
        if len(client_deltas) == 0:
            break
        
        # Compute normalization factor: sum of all tau_i
        total_tau = sum(tau for _, tau in client_deltas)
        
        # Normalized weighted average of deltas
        weighted_delta = [np.zeros_like(w) for w in global_weights]
        for delta, tau_i in client_deltas:
            for i, d in enumerate(delta):
                weighted_delta[i] += (tau_i / total_tau) * d
        
        # Update global weights: w_{t+1} = w_t - weighted_delta
        global_weights = [w - d for w, d in zip(global_weights, weighted_delta)]

        # extract just the deltas for metrics function
        deltas_only = [delta for (delta, _tau) in client_deltas]
        metrics, prev_acc, prev_global_weights, comm_cost = compute_round_metrics(
            "FedNova",
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
            client_deltas=deltas_only,
        )
        if callable(report):
            report("FedNova", rnd + 1, metrics)
    
    # Final evaluation
    global_model.set_weights(global_weights)
    loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
    return float(acc)

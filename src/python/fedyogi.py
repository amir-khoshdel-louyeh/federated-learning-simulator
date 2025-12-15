import time
import numpy as np
from model import make_model
from metrics import compute_round_metrics


def fedyogi(x_train, y_train, x_test, y_test, clients=3, rounds=3, local_epochs=1,
            batch_size=64, server_lr=0.01, beta1=0.9, beta2=0.999, tau=1e-3,
            full_data_per_client=False, report=None):
    """FedYogi: Adaptive Federated Optimization with server-side Yogi optimizer.
    
    Similar to FedAdam but uses a different second moment update rule:
    v_t = v_{t-1} - (1 - beta2) * sign(v_{t-1} - grad^2) * grad^2
    
    This makes it more robust to noisy gradients compared to Adam.
    
    Args:
        server_lr: Server learning rate
        beta1: Exponential decay rate for first moment estimates
        beta2: Exponential decay rate for second moment estimates
        tau: Small constant for numerical stability
    
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
    
    # Initialize moments (server-side)
    m_t = [np.zeros_like(w) for w in global_weights]  # First moment
    v_t = [np.zeros_like(w) for w in global_weights]  # Second moment
    
    for t in range(1, rounds + 1):
        round_start = time.perf_counter()
        client_weights = []
        
        for shard in shards:
            if len(shard) == 0:
                continue
            
            local = make_model()
            local.set_weights(global_weights)
            local.fit(x_train[shard], y_train[shard], epochs=local_epochs,
                     batch_size=batch_size, verbose=0)
            client_weights.append(local.get_weights())
        
        if len(client_weights) == 0:
            break
        
        # Average client weights
        avg_weights = []
        for weights in zip(*client_weights):
            avg_weights.append(np.mean(weights, axis=0))
        
        # Compute pseudo-gradient
        delta = [w_old - w_new for w_old, w_new in zip(global_weights, avg_weights)]
        
        # Update moments and weights using Yogi
        new_global_weights = []
        for i, (w, d, m, v) in enumerate(zip(global_weights, delta, m_t, v_t)):
            # Update first moment (same as Adam)
            m_new = beta1 * m + (1 - beta1) * d
            
            # Update second moment (Yogi-style: adaptive with sign)
            v_new = v - (1 - beta2) * np.sign(v - d ** 2) * (d ** 2)
            
            # Bias correction
            m_hat = m_new / (1 - beta1 ** t)
            v_hat = v_new / (1 - beta2 ** t)
            
            # Update weights
            w_new = w - server_lr * m_hat / (np.sqrt(np.abs(v_hat)) + tau)
            
            new_global_weights.append(w_new)
            m_t[i] = m_new
            v_t[i] = v_new
        
        global_weights = new_global_weights

        metrics, prev_acc, prev_global_weights, comm_cost = compute_round_metrics(
            "FedYogi",
            t,
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
            reference_weights=avg_weights,
        )
        if callable(report):
            report("FedYogi", t, metrics)
    
    # Final evaluation
    global_model.set_weights(global_weights)
    loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
    return float(acc)

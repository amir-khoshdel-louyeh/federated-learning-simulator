import numpy as np
from model import make_model


def fedadam(x_train, y_train, x_test, y_test, clients=3, rounds=3, local_epochs=1, 
            batch_size=64, server_lr=0.01, beta1=0.9, beta2=0.999, tau=1e-3, 
            full_data_per_client=False):
    """FedAdam: Adaptive Federated Optimization with server-side Adam.
    
    Uses momentum (beta1) and adaptive learning rates (beta2) on the server
    to update the global model based on the pseudo-gradient (difference between
    current and averaged client models).
    
    Args:
        server_lr: Server learning rate (eta in paper)
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
    
    # Initialize Adam moments (server-side)
    m_t = [np.zeros_like(w) for w in global_weights]  # First moment
    v_t = [np.zeros_like(w) for w in global_weights]  # Second moment
    
    for t in range(1, rounds + 1):
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
        
        # Compute pseudo-gradient: delta = w_t - avg_weights
        delta = [w_old - w_new for w_old, w_new in zip(global_weights, avg_weights)]
        
        # Update moments and global weights using Adam
        new_global_weights = []
        for i, (w, d, m, v) in enumerate(zip(global_weights, delta, m_t, v_t)):
            # Update biased first moment estimate
            m_new = beta1 * m + (1 - beta1) * d
            # Update biased second moment estimate
            v_new = beta2 * v + (1 - beta2) * (d ** 2)
            
            # Bias correction
            m_hat = m_new / (1 - beta1 ** t)
            v_hat = v_new / (1 - beta2 ** t)
            
            # Update weights
            w_new = w - server_lr * m_hat / (np.sqrt(v_hat) + tau)
            
            new_global_weights.append(w_new)
            m_t[i] = m_new
            v_t[i] = v_new
        
        global_weights = new_global_weights
    
    # Final evaluation
    global_model.set_weights(global_weights)
    loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
    return float(acc)

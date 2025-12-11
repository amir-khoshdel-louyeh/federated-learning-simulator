import numpy as np
from model import make_model


def fedadagrad(x_train, y_train, x_test, y_test, clients=3, rounds=3, local_epochs=1,
               batch_size=64, server_lr=0.01, tau=1e-3, full_data_per_client=False):
    """FedAdagrad: Adaptive Federated Optimization with server-side Adagrad.
    
    Uses accumulated squared gradients on the server to adapt learning rates
    per parameter. Unlike Adam/Yogi, doesn't use exponential moving average,
    so learning rate decreases monotonically.
    
    Args:
        server_lr: Server learning rate
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
    
    # Initialize accumulated squared gradients (server-side)
    v_t = [np.zeros_like(w) for w in global_weights]
    
    for t in range(rounds):
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
        
        # Update accumulated squared gradients and weights
        new_global_weights = []
        for i, (w, d, v) in enumerate(zip(global_weights, delta, v_t)):
            # Accumulate squared gradients
            v_new = v + d ** 2
            
            # Update weights
            w_new = w - server_lr * d / (np.sqrt(v_new) + tau)
            
            new_global_weights.append(w_new)
            v_t[i] = v_new
        
        global_weights = new_global_weights
    
    # Final evaluation
    global_model.set_weights(global_weights)
    loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
    return float(acc)

import time
from typing import List, Optional, Tuple, Dict, Any

import numpy as np


def weight_bytes(weights: List[np.ndarray]) -> int:
    return sum(np.array(w).nbytes for w in weights)


def communication_cost_update(prev_cost: int, clients: int, weights: List[np.ndarray]) -> int:
    # simple approximation: each round, each client uploads and downloads the full weights
    bytes_per_round = clients * 2 * weight_bytes(weights)
    return prev_cost + bytes_per_round


def stability_variance_from_weights(local_weights_list: List[List[np.ndarray]], reference_weights: List[np.ndarray]) -> float:
    if not local_weights_list:
        return 0.0
    norms = []
    for w_local in local_weights_list:
        norm_sum = 0.0
        for wl, wr in zip(w_local, reference_weights):
            diff = wl - wr
            norm_sum += float(np.linalg.norm(diff))
        norms.append(norm_sum)
    return float(np.var(norms))


def stability_variance_from_deltas(client_deltas: List[List[np.ndarray]]) -> float:
    if not client_deltas:
        return 0.0
    norms = []
    for delta in client_deltas:
        norm_sum = 0.0
        for d in delta:
            norm_sum += float(np.linalg.norm(d))
        norms.append(norm_sum)
    return float(np.var(norms))


def velocity(prev_global_weights: List[np.ndarray], global_weights: List[np.ndarray]) -> float:
    vel_sum = 0.0
    for w_prev, w_cur in zip(prev_global_weights, global_weights):
        vel_sum += float(np.linalg.norm(w_cur - w_prev))
    return vel_sum


def compute_round_metrics(
    algo_name: str,
    round_index: int,
    start_time: float,
    global_model,
    x_test,
    y_test,
    prev_acc: Optional[float],
    prev_global_weights: List[np.ndarray],
    global_weights: List[np.ndarray],
    clients: int,
    comm_cost: int,
    local_weights_list: Optional[List[List[np.ndarray]]] = None,
    reference_weights: Optional[List[np.ndarray]] = None,
    client_deltas: Optional[List[List[np.ndarray]]] = None,
) -> Tuple[Dict[str, Any], float, List[np.ndarray], int]:
    # Evaluate accuracy
    global_model.set_weights(global_weights)
    loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
    # Convergence
    convergence = 0.0 if prev_acc is None else (acc - prev_acc)
    # Stability / Variance
    if local_weights_list is not None and reference_weights is not None:
        stability_var = stability_variance_from_weights(local_weights_list, reference_weights)
    elif client_deltas is not None:
        stability_var = stability_variance_from_deltas(client_deltas)
    else:
        stability_var = 0.0
    # Velocity
    vel = velocity(prev_global_weights, global_weights) if prev_global_weights else 0.0
    # Communication cost (cumulative)
    comm_cost = communication_cost_update(comm_cost, clients, global_weights)
    # Training time
    train_time = time.perf_counter() - start_time

    metrics = {
        "accuracy": float(acc),
        "convergence": float(convergence),
        "communication_cost": int(comm_cost),
        "stability_variance": float(stability_var),
        "training_time": float(train_time),
        "velocity": float(vel),
    }

    return metrics, float(acc), [np.copy(w) for w in global_weights], comm_cost

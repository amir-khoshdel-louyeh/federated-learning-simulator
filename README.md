# Federated Learning Simulator

This project provides a simulator for various federated learning algorithms, allowing users to experiment with and compare different approaches to distributed machine learning. The simulator includes implementations of several popular algorithms, a simple GUI, and utilities for data handling and aggregation.

## Algorithms Implemented

### FedAvg
Federated Averaging (FedAvg) is a foundational algorithm in federated learning. It works by having each client train a local model on its own data and then sending the model updates to a central server, which averages the updates to form a new global model. This process is repeated over multiple rounds, enabling collaborative learning without sharing raw data.

### FedSGD
Federated Stochastic Gradient Descent (FedSGD) is a variant where clients compute gradients on their local data and send these gradients to the server, which aggregates them to update the global model. Unlike FedAvg, which averages model weights, FedSGD averages gradients, making it suitable for synchronous updates.

### FedProx
FedProx extends FedAvg by introducing a proximal term in the local objective function to address heterogeneity among client data. This term penalizes local updates that deviate too much from the global model, improving convergence and stability when client data distributions are non-IID (not independent and identically distributed).

### FedOpt
Federated Optimization (FedOpt) generalizes FedAvg by allowing the server to use advanced optimization algorithms (such as Adam or Yogi) instead of simple averaging. This can accelerate convergence and improve performance, especially in challenging federated settings.

### FedPer
Federated Personalization (FedPer) is designed to balance global knowledge sharing with local personalization. It allows each client to maintain a personalized part of the model while sharing a common global part, enabling better adaptation to local data characteristics.

### Secure Aggregation
Secure Aggregation is a cryptographic protocol that ensures the privacy of client updates during aggregation. It allows the server to compute the sum (or average) of client updates without learning any individual client's contribution, enhancing privacy in federated learning.

### Split Learning
Split Learning is a collaborative learning approach where the model is split between clients and the server. Clients train the initial layers and send intermediate representations to the server, which completes the forward and backward passes. This reduces client-side computation and can improve privacy.

## Project Structure
- `src/`: Source code for algorithms, GUI, and utilities
- `data/`: Datasets used for training and evaluation
- `output/`: Output files and results
- `requirements.txt`: Python dependencies

## Getting Started
To keep your dependencies isolated, it is recommended to use a Python virtual environment. You can create and activate one as follows:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the simulator GUI:
   ```bash
   python src/gui.py
   ```
3. Select an algorithm and dataset to start experimenting.

## License
This project is licensed under the MIT License.
# Federated Learning Simulator: Code and Algorithm Explanations

This document explains all classes and functions in `src/fedavg_simulator.py`, in the same order as the code. Each section includes the signature, purpose, and a clear explanation of how it works.

---

## `class SimpleCNN(nn.Module)`
**Purpose:**
A simple convolutional neural network for MNIST digit classification.

**How it works:**
- Two convolutional layers (with ReLU and max pooling) extract features from $28\times28$ grayscale images.
- The output is flattened and passed through two fully connected layers to produce class logits.
- Used as the model architecture for all clients and the global model.

---

## `def iid_split(dataset, num_clients)`
**Purpose:**
Splits a dataset randomly and evenly among all clients (IID: independent and identically distributed).

**How it works:**
- Shuffles all data indices.
- Divides indices into equal chunks for each client.
- Any leftover samples are given to the last client.
- Returns a dictionary mapping client index to list of data indices.

---

## `def non_iid_split(dataset, num_clients, num_shards=200, shards_per_client=2)`
**Purpose:**
Splits a dataset into non-IID partitions, so each client gets data biased toward certain labels.

**How it works:**
- Sorts all data indices by label.
- Divides sorted indices into many small shards.
- Randomly assigns a few shards to each client (so each client sees only a subset of labels).
- Distributes any leftover shards round-robin.
- Returns a dictionary mapping client index to list of data indices.

---

## `def get_dataloaders(num_clients=4, batch_size=32, quick=False, non_iid=False)`
**Purpose:**
Prepares PyTorch dataloaders for each client and the test set, supporting both IID and non-IID splits and a quick mode for fast testing.

**How it works:**
- Loads MNIST train and test sets.
- If `quick` is True, uses small subsets for smoke testing.
- Splits training data among clients using IID or non-IID strategy.
- Returns:
  - `client_loaders`: dict of DataLoader for each client
  - `test_loader`: DataLoader for test set
  - `client_data_lens`: dict of number of samples per client

---

## `def train_local(model, dataloader, epochs, lr=0.01, device='cpu')`
**Purpose:**
Trains a model on a client's local data for a fixed number of epochs.

**How it works:**
- Sets model to training mode and moves it to the correct device.
- Uses cross-entropy loss and SGD optimizer.
- For each epoch, iterates over batches:
  - Moves data to device
  - Forward pass, computes loss
  - Backward pass, updates parameters
- Returns the updated model parameters (state dict).

---

## `def evaluate(model, dataloader, device='cpu')`
**Purpose:**
Evaluates a model's accuracy on a given dataset.

**How it works:**
- Sets model to evaluation mode and moves it to the correct device.
- Disables gradient computation.
- For each batch:
  - Moves data to device
  - Computes predictions
  - Counts correct predictions
- Returns accuracy as a float (fraction correct).

---

## `def fedavg_aggregate(global_model, client_states, client_lens)`
**Purpose:**
Aggregates model parameters from multiple clients using a weighted average, where each client's contribution is proportional to its number of training samples (FedAvg algorithm).

**How it works:**
- For each parameter in the model, computes a weighted sum across all clients:
  - $w_{global} = \sum_{k=1}^K \frac{n_k}{N} w_k$
    - $K$: number of clients
    - $n_k$: number of samples on client $k$
    - $N$: total samples across all clients
    - $w_k$: parameter tensor from client $k$
- Initializes an empty accumulator for each parameter, then adds each client's parameter tensor multiplied by its weight.
- Updates the global model with the aggregated parameters.

---

## `def parse_args()`
**Purpose:**
Parses command-line arguments for the simulator.

**How it works:**
- Uses `argparse` to define and parse options for number of clients, rounds, local epochs, batch size, client fraction, non-IID mode, quick mode, and random seed.
- Returns the parsed arguments.

---

## `def main()`
**Purpose:**
Runs the full federated learning simulation.

**How it works:**
- Parses arguments and sets random seeds for reproducibility.
- Selects device (CPU or GPU).
- Prepares dataloaders for clients and test set.
- Initializes the global model.
- For each round:
  - Optionally samples a fraction of clients.
  - Broadcasts the global model to sampled clients.
  - Each client trains locally and returns updated parameters.
  - Aggregates updates using FedAvg.
  - Evaluates the global model and records accuracy.
- After all rounds:
  - Saves accuracy plots and CSV file with results.

---

## `if __name__ == '__main__': main()`
**Purpose:**
Entry point for running the script directly.

**How it works:**
- Calls `main()` if the script is executed as the main program.

---

## Example Commands

Install dependencies and run a quick test:

```bash
pip install -r requirements.txt
python src/fedavg_simulator.py --clients 2 --rounds 20 --local-epochs 1 --quick
```

Run a full experiment:

```bash
python src/fedavg_simulator.py --clients 4 --rounds 50 --local-epochs 5
```

---

For further extensions (unit tests, optimizer options, IID vs Non-IID comparison), see the project README or request specific features.
118

Federated Learning Simulator (FedAvg)

This is a small federated learning simulation implemented in Python using PyTorch.

Features
- Uses MNIST by default (small dataset, easy to run).
- Splits dataset across simulated clients (IID or Non-IID).
- Each client trains a local neural network.
- Server aggregates client weights using FedAvg.
- Evaluates the global model after each communication round.
- Saves an accuracy-vs-round plot to `output/accuracy.png`.

Quick start (after creating a venv):

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Run a quick smoke test (small subset, 2 clients, 1 round):

```bash
python src/fedavg_simulator.py --clients 2 --rounds 1 --local-epochs 1 --quick
```

Options
- `--clients`: number of simulated clients
- `--rounds`: communication rounds
- `--local-epochs`: epochs each client trains per round
- `--non-iid`: enable Non-IID split
- `--frac`: fraction of clients sampled each round (default 1.0)
- `--quick`: use small subsets for fast smoke tests

Files
- `src/fedavg_simulator.py`: main simulator script

Notes
- Uses CPU by default. Training can be slow with many rounds/clients.
- For experimentation with Non-IID splits, run with `--non-iid` and try different client counts.

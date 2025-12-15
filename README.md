Federated Learning Simulator

This project provides a simple GUI-based simulator for centralized and federated training on MNIST using TensorFlow/Keras.

Features
- MNIST loader with configurable train/val/test sizes.
- Centralized training baseline.
- Federated algorithms: FedAvg, FedProx, SCAFFOLD, FedAdam, FedYogi, FedAdagrad, FedNova.
- Per-round metrics tracked: Accuracy, Convergence, Communication Cost, Stability/Variance, Training Time, Velocity.
- Result persistence to `result.txt` and embedded comparison graphs inside the GUI.

Quick start

1) Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2) Run the GUI

```bash
python main.py
```

Usage
- Enter total dataset size; the app splits 80/10/10 into train/val/test.
- Choose number of clients and rounds.
- Optionally give full dataset to each client.
- Select algorithms and click "Run selected algorithms".
- After training, results are saved to `result.txt` and comparison plots are shown in a scrollable window.

Project structure
- `main.py`: GUI and orchestration.
- `model.py`: MNIST loader and model factory.
- `centralized.py`: Centralized baseline.
- `federated_average.py`, `fedprox.py`, `scaffold.py`, `fedadam.py`, `fedyogi.py`, `fedadagrad.py`, `fednova.py`: Federated algorithms.
- `metrics.py`: Common metric computations.
- `comparison.py`: Functions to generate comparison figures from `result.txt`.


Notes
- Training runs on CPU by default; set `CUDA_VISIBLE_DEVICES` accordingly if using GPU.
- TensorFlow downloads MNIST on first run.

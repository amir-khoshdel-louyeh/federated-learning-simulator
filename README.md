# MNIST Centralized vs Federated Demo (Tkinter)

This is a minimal Python program demonstrating centralized training vs a simple simulated federated averaging workflow using the MNIST dataset and a Tkinter GUI.

What it does:
- Loads a small fixed subset of MNIST (default 3000 samples) and a small test set (default 1000 samples).
- Trains a simple neural network centrally and reports test accuracy.
- Simulates federated learning across 3 clients by splitting the same training data, training locally, averaging weights, and reporting test accuracy.

Files added:
- `main.py`: the Tkinter GUI and the training/simulation code.
- `requirements.txt`: minimal dependencies.

Requirements
------------
- Python 3.8+
- TensorFlow (the example uses `tf.keras.datasets.mnist`; installing `tensorflow` via pip is required)
- numpy
- Tkinter (usually included with Python; on some Linux distributions you may need `python3-tk`)

Install (recommended in a virtualenv):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run
---

```bash
python main.py
```

Usage notes
-----------
- Click "Load MNIST subset" to fetch and prepare the data (first run will download the dataset).
- Click "Train Centralized" to train on the full subset and see the test accuracy.
- Click "Run Federated" to run the federated averaging simulation and compare test accuracy.
- You can adjust the subset size, number of federated rounds, and local/central epochs in the UI.

This program is intentionally simple and small to illustrate the concept rather than deliver production performance.

If you don't want to install TensorFlow, you can adapt the code to use scikit-learn or a lighter-weight library, but that is outside the scope of this demo.

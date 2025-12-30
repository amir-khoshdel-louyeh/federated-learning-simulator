````markdown
# Federated Learning Simulator

A simple simulator for centralized and federated training on MNIST with both Python (TensorFlow/Keras) and pure-MATLAB implementations. It tracks key metrics per round and can compare Python vs MATLAB runs side-by-side.

## Features

- MNIST loader with configurable train/val/test sizes (80/10/10 split by default)
- Dataset selection in GUI: IID (MNIST) or Non-IID (Fashion-MNIST)
- Non-IID options: configurable common label (e.g., Sneaker) and shared fraction
- Centralized baseline and federated algorithms:
	- FedAvg, FedProx, SCAFFOLD, FedAdam, FedYogi, FedAdagrad, FedNova
- Per-round metrics:
	- Accuracy, Convergence, Communication Cost, Stability / Variance, Training Time, Velocity
- Results saved to text files for easy cross-language comparison
- Comparison tool that renders:
	- Python-only, MATLAB-only, and combined overlay plots (when both results exist)
		- Figures are automatically saved to `src/graphs/`

## Repository structure

```
src/
	python/
		main.py                # GUI app (Tkinter)
		comparison.py          # Figure generator (supports Python, MATLAB, or both)
		model.py               # Dataset loaders (MNIST, Fashion-MNIST) + model factory
		partition.py           # Non-IID shard builders (primary label + common portion)
		metrics.py             # Common metric computations
		centralized.py         # Centralized baseline
		fedavg.py   		   # FedAvg
		fedprox.py             # FedProx
		scaffold.py            # SCAFFOLD
		fedadam.py             # FedAdam (server-side Adam)
		fedyogi.py             # FedYogi (server-side Yogi)
		fedadagrad.py          # FedAdagrad (server-side Adagrad)
		fednova.py             # FedNova
		requirements.txt       # Python dependencies

	matlab/
		main.m                 # Prompts inputs, runs ALL algorithms, writes results
		load_mnist.m           # Downloads/loads MNIST (IDX) into MATLAB arrays
		make_model.m           # Simple MLP with manual backprop + Adam
		compute_round_metrics.m# Metric computations (MATLAB)
		make_shards.m          # Client data partition helper
		*.m                    # Algorithm files (FedAvg, FedProx, etc.)

	results/
		p_result.txt           # Python results (written by GUI)
		m_result.txt           # MATLAB results (written by main.m)
		result.txt             # Legacy filename for compatibility
```

## Results files

- Python GUI writes: `src/results/p_result.txt` (and also `src/results/result.txt` for legacy tools).
- MATLAB runner writes: `src/results/m_result.txt`.
- Legacy note: older versions may have written to a top-level `results/` folder. Current tools look in `src/results/`. If you have legacy files in `results/`, move them into `src/results/` to be picked up by the comparison tool.

Format: one line per algorithm in the form

```
<algo_key>={<Python-like-dict-with-lists>}
```

Example (truncated):

```
fedavg={'Accuracy': [0.81, 0.87, 0.88], 'Convergence': [0.0, 0.06, 0.01], ...}
```

These files are parsed with Python’s `ast.literal_eval`, so keys and list types should remain as written.

## Python (GUI) — how to run

1) Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r src/python/requirements.txt
```

2) Launch the GUI

```bash
python src/python/main.py
```

Usage in the GUI:
- Enter total dataset size; app splits 80/10/10 into train/val/test (live preview displayed)
- Choose number of clients and rounds
- Choose data distribution: IID (MNIST) or Non-IID (Fashion-MNIST)
- If Non-IID: configure common label and shared fraction in the options panel
- Select algorithms and click “Run selected algorithms”
- When finished, the app writes `src/results/p_result.txt` and displays comparison graphs

## MATLAB — how to run

In MATLAB or MATLAB Online (from the repository root):

```matlab
addpath(fullfile(pwd, 'src', 'matlab'))
main
```

You’ll be prompted for:
- Total dataset size
- Number of clients
- Number of rounds
- Whether each client gets the full dataset or partitions are split among clients

Note: The Python GUI no longer includes a “full dataset per client” toggle. Non-IID behavior in Python is achieved via Fashion-MNIST with shard builders in `src/python/partition.py`.

The script runs ALL algorithms and writes `src/results/m_result.txt`.

Notes:
- MNIST is downloaded on first run to `src/matlab/data/` (relative to the MATLAB files).
- The model is a small MLP (Flatten → Dense(64, ReLU) → Dense(10, Softmax)) with manual backprop and Adam updates (no toolboxes required).
- `main.m` automatically adds its `algorithms` subfolder to the MATLAB path, so functions like `fedavg`, `fedprox`, etc., resolve in MATLAB Online.

Troubleshooting:
- If you see "Unrecognized function or variable 'fedavg'", ensure your session has added `src/matlab` to the path (use the `addpath` line above) and then re-run `main`.

## Comparing results and generating graphs

The comparison tool supports three modes automatically:
- Only Python results present → one figure per metric for Python
- Only MATLAB results present → one figure per metric for MATLAB
- Both present → for each metric, three figures in order:
	1) Python-only (p_result)
	2) MATLAB-only (m_result)
	3) Combined overlay (series tagged with “(Py)” and “(Mat)”) 

Run standalone from the shell:

```bash
python src/python/comparison.py
```

Or view embedded in the GUI after a Python run; it will look in `src/results/` for `p_result.txt` and `m_result.txt`.

All generated figures are also saved under `src/graphs/` with filenames like:

- `YYYYMMDD-HHMMSS__Accuracy__python.png`
- `YYYYMMDD-HHMMSS__Accuracy__matlab.png`
- `YYYYMMDD-HHMMSS__Accuracy__both.png`

## Tips and troubleshooting

- If you move or rename the `results/` folder, update your paths accordingly.
- If running Python on a headless server and `matplotlib` cannot show windows, you can still run `comparison.py`; it will create figures. To save figures to files instead of showing them, open an issue and we’ll add a small exporter.
- If MATLAB can’t find functions, ensure your current folder is `src/matlab` or call `addpath('src/matlab')`.
- Python dependencies are pinned in `src/python/requirements.txt`.

## License

MIT (see LICENSE if present). Contributions welcome.
````

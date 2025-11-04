# Federated Learning Simulator

This project provides a simulator for various federated learning aggregation algorithms, allowing users to experiment with and compare different approaches to distributed machine learning. The simulator includes implementations of 7 popular algorithms, a simple GUI, and utilities for data handling and partitioning.

## Algorithms Implemented

The following aggregation algorithms are available in the `src/aggregation_algorithms/` directory:

1. **FedAvg** - Federated Averaging (baseline)
2. **FedProx** - Proximal term for heterogeneous data
3. **FedOpt** - Server-side adaptive optimization
4. **FedNova** - Normalized averaging by training steps
5. **SCAFFOLD** - Control variates for drift correction
6. **Clustered FL** - Separate models per client cluster
7. **Personalized FL** - Global training + local fine-tuning

For detailed documentation on each algorithm, see [`docs/ALGORITHMS.md`](docs/ALGORITHMS.md).

## Project Structure
- `src/aggregation_algorithms/` - Federated learning algorithm implementations
- `src/` - Core utilities (data loading, partitioning, GUI)
- `docs/` - Algorithm documentation
- `requirements.txt` - Python dependencies

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

3. **New GUI Features:**
    - **Data Partitioning:**
       - Choose between `IID` (default) and `non-IID` data partitioning for clients. Non-IID simulates more realistic federated settings where each client may have data from only a subset of classes.
    - **Run All Algorithms:**
       - Use the `Run All Algorithms` button to execute all 7 implemented algorithms (FedAvg, FedOpt, FedProx, FedNova, SCAFFOLD, Clustered FL, Personalized FL) with the selected parameters and partitioning strategy.
    - **How to use:**
       1. Select the desired algorithm or choose `Run All Algorithms`.
       2. Set the number of clients, rounds, local epochs, and batch size.
       3. Select the data partitioning strategy (IID or non-IID).
       4. Click `Run` or `Run All Algorithms` to start.

## License
This project is licensed under the MIT License.
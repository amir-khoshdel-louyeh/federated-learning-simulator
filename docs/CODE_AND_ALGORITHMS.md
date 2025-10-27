# Code and Algorithms — line-by-line

This file contains an exact, line-numbered copy of `src/fedavg_simulator.py` followed by detailed explanations of the algorithms used (FedAvg, IID/Non-IID splitting, training loop, aggregation, and outputs).

---

## Exact source: `src/fedavg_simulator.py` (line-by-line)

1	import argparse
2	import copy
3	import os
4	import random
5	from collections import defaultdict
6
7	import matplotlib.pyplot as plt
8	import numpy as np
9	import torch
10	import torch.nn as nn
11	import torch.optim as optim
12	import torchvision
13	import torchvision.transforms as transforms
14	from torch.utils.data import DataLoader, Subset
15
16
17	# Simple CNN for MNIST
18	class SimpleCNN(nn.Module):
19	    def __init__(self, num_classes=10):
20	        super(SimpleCNN, self).__init__()
21	        self.features = nn.Sequential(
22	            nn.Conv2d(1, 32, 3, 1, 1),
23	            nn.ReLU(inplace=True),
24	            nn.MaxPool2d(2),
25	            nn.Conv2d(32, 64, 3, 1, 1),
26	            nn.ReLU(inplace=True),
27	            nn.MaxPool2d(2),
28	        )
29	        self.classifier = nn.Sequential(
30	            nn.Flatten(),
31	            nn.Linear(64 * 7 * 7, 128),
32	            nn.ReLU(inplace=True),
33	            nn.Linear(128, num_classes),
34	        )
35
36	    def forward(self, x):
37	        x = self.features(x)
38	        x = self.classifier(x)
39	        return x
40
41
42	def iid_split(dataset, num_clients):
43	    # Even split of indices
44	    num_items = len(dataset) // num_clients
45	    all_idxs = list(range(len(dataset)))
46	    random.shuffle(all_idxs)
47	    dict_clients = {i: all_idxs[i * num_items: (i + 1) * num_items] for i in range(num_clients)}
48	    # give remainder to last client
49	    rest = all_idxs[num_clients * num_items:]
50	    if rest:
51	        dict_clients[num_clients - 1].extend(rest)
52	    return dict_clients
53
54
55	def non_iid_split(dataset, num_clients, num_shards=200, shards_per_client=2):
56	    # Following a common non-iid split from literature: sort by label and divide into shards
57	    targets = np.array(dataset.targets)
58	    idxs = np.arange(len(targets))
59	    # sort by label
60	    idxs_labels = np.vstack((idxs, targets)).T
61	    idxs_labels = idxs_labels[idxs_labels[:, 1].argsort()]
62	    idxs = idxs_labels[:, 0].astype(int)
63	    shard_size = len(dataset) // num_shards
64	    shards = [list(idxs[i * shard_size: (i + 1) * shard_size]) for i in range(num_shards)]
65	    random.shuffle(shards)
66	    dict_clients = {i: [] for i in range(num_clients)}
67	    for i in range(num_clients):
68	        assigned = []
69	        for _ in range(shards_per_client):
70	            if shards:
71	                assigned.extend(shards.pop())
72	        dict_clients[i] = assigned
73	    # If shards remain, distribute
74	    si = 0
75	    while shards:
76	        dict_clients[si % num_clients].extend(shards.pop())
77	        si += 1
78	    return dict_clients
79
80
81	def get_dataloaders(num_clients=4, batch_size=32, quick=False, non_iid=False):
82	    transform = transforms.Compose([transforms.ToTensor()])
83	    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
84	    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
85
86	    if quick:
87	        # Use only a fraction for smoke tests
88	        train_subset = list(range(2000))
89	        test_subset = list(range(1000))
90	        trainset = Subset(trainset, train_subset)
91	        testset = Subset(testset, test_subset)
92	        # For Subset, targets attribute is not available; hack: wrap
93	        class QuickDataset(Subset):
94	            @property
95	            def targets(self):
96	                return [trainset.dataset.targets[i] for i in self.indices]
97
98	        trainset = QuickDataset(trainset.dataset, trainset.indices)
99	        testset = Subset(testset.dataset, testset.indices)
100
101	    if non_iid:
102	        client_idxs = non_iid_split(trainset if not quick else trainset, num_clients)
103	    else:
104	        client_idxs = iid_split(trainset, num_clients)
105
106	    client_loaders = {}
107	    client_data_lens = {}
108	    for i in range(num_clients):
109	        idxs = client_idxs[i]
110	        subset = Subset(trainset, idxs)
111	        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
112	        client_loaders[i] = loader
113	        client_data_lens[i] = len(subset)
114
115	    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
116	    return client_loaders, test_loader, client_data_lens
117
118
119	def train_local(model, dataloader, epochs, lr=0.01, device='cpu'):
120	    model.train()
121	    model.to(device)
122	    criterion = nn.CrossEntropyLoss()
123	    optimizer = optim.SGD(model.parameters(), lr=lr)
124	    for e in range(epochs):
125	        for data, target in dataloader:
126	            data, target = data.to(device), target.to(device)
127	            optimizer.zero_grad()
128	            output = model(data)
129	            loss = criterion(output, target)
130	            loss.backward()
131	            optimizer.step()
132	    return model.state_dict()
133
134
135	def evaluate(model, dataloader, device='cpu'):
136	    model.eval()
137	    model.to(device)
138	    correct = 0
139	    total = 0
140	    with torch.no_grad():
141	        for data, target in dataloader:
142	            data, target = data.to(device), target.to(device)
143	            outputs = model(data)
144	            _, preds = torch.max(outputs, 1)
145	            correct += (preds == target).sum().item()
146	            total += target.size(0)
147	    return correct / total if total > 0 else 0.0
148
149
150	def fedavg_aggregate(global_model, client_states, client_lens):
151	    # Weighted average by number of samples per client
152	    # client_lens is a dict mapping client_id -> num_samples
153	    total_samples = sum(client_lens.values())
154	    global_dict = global_model.state_dict()
155	    # initialize
156	    for k in global_dict.keys():
157	        global_dict[k] = torch.zeros_like(global_dict[k])
158	    for client_id, state in client_states.items():
159	        weight = client_lens[client_id] / total_samples
160	        for k in global_dict.keys():
161	            global_dict[k] += state[k] * weight
162	    global_model.load_state_dict(global_dict)
163	    return global_model
164
165
166	def parse_args():
167	    parser = argparse.ArgumentParser(description='Federated Learning Simulator (FedAvg)')
168	    parser.add_argument('--clients', type=int, default=4)
169	    parser.add_argument('--rounds', type=int, default=5)
170	    parser.add_argument('--local-epochs', type=int, default=1)
171	    parser.add_argument('--batch-size', type=int, default=32)
172	    parser.add_argument('--frac', type=float, default=1.0, help='fraction of clients sampled each round')
173	    parser.add_argument('--non-iid', dest='non_iid', action='store_true')
174	    parser.add_argument('--quick', action='store_true', help='use small subset for quick smoke test')
175	    parser.add_argument('--seed', type=int, default=42)
176	    return parser.parse_args()
177
178
179	def main():
180	    args = parse_args()
181	    random.seed(args.seed)
182	    np.random.seed(args.seed)
183	    torch.manual_seed(args.seed)
184
185	    device = 'cuda' if torch.cuda.is_available() else 'cpu'
186	    print(f"Using device: {device}")
187
188	    client_loaders, test_loader, client_lens = get_dataloaders(num_clients=args.clients, batch_size=args.batch_size, quick=args.quick, non_iid=args.non_iid)
189
190	    # Initialize global model
191	    global_model = SimpleCNN()
192
193	    acc_history = []
194
195	    for r in range(args.rounds):
196	        print(f"\n--- Round {r+1}/{args.rounds} ---")
197	        sampled_clients = list(range(args.clients))
198	        if args.frac < 1.0:
199	            sample_size = max(1, int(args.frac * args.clients))
200	            sampled_clients = random.sample(range(args.clients), sample_size)
201	        print(f"Sampled clients this round: {sampled_clients}")
202
203	        # broadcast
204	        global_state = copy.deepcopy(global_model.state_dict())
205
206	        client_states = {}
207	        # each client trains locally
208	        for cid in sampled_clients:
209	            local_model = SimpleCNN()
210	            local_model.load_state_dict(global_state)
211	            print(f"Client {cid}: training on {client_lens[cid]} samples")
212	            state = train_local(local_model, client_loaders[cid], epochs=args.local_epochs, device=device)
213	            client_states[cid] = state
214
215	        # aggregate
216	        global_model = fedavg_aggregate(global_model, client_states, client_lens)
217
218	        # evaluate
219	        acc = evaluate(global_model, test_loader, device=device)
220	        acc_history.append(acc)
221	        print(f"Global model accuracy after round {r+1}: {acc*100:.2f}%")
222
223	    # Save plot and accuracy data
224	    os.makedirs('output', exist_ok=True)
225	    rounds = list(range(1, len(acc_history) + 1))
226	    acc_percent = [a * 100 for a in acc_history]
227
228	    plt.figure()
229	    plt.plot(rounds, acc_percent, marker='o')
230	    plt.xlabel('Round')
231	    plt.ylabel('Test Accuracy (%)')
232	    plt.title('FedAvg: Global Model Accuracy per Round')
233	    plt.grid(True)
234
235	    # Save a basic plot
236	    plt.savefig('output/accuracy.png')
237
238	    # Annotate points and save a higher-resolution copy
239	    for x, y in zip(rounds, acc_percent):
240	        plt.annotate(f'{y:.2f}%', (x, y), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
241	    plt.tight_layout()
242	    plt.savefig('output/accuracy_annotated.png', dpi=200)
243
244	    # Save raw numbers to accuracy.txt (round, accuracy_percent)
245	    txt_path = os.path.join('output', 'accuracy.txt')
246	    with open(txt_path, 'w') as f:
247	        f.write('round,accuracy_percent\n')
248	        for x, y in zip(rounds, acc_percent):
249	            f.write(f"{x},{y:.4f}\n")
250
251	    print('\nSaved accuracy plot to output/accuracy.png')
252	    print('Saved annotated plot to output/accuracy_annotated.png')
253	    print(f'Saved accuracy data to {txt_path}')
254
255
256	if __name__ == '__main__':
257	    main()
258
---

## Algorithmic details and explanations

### 1) FedAvg (server-side aggregation)

FedAvg is implemented in `fedavg_aggregate()` (lines 150-163). The mathematical formula used is the weighted average of client model parameters:

- Notation:
  - K : number of participating clients
  - n_k : number of samples on client k
  - N = sum_k n_k : total number of samples across participating clients
  - w_k : parameter vector (state dict) from client k after local training

- Aggregation formula (applied per-parameter):

  w_global = sum_{k=1..K} (n_k / N) * w_k

Implementation notes:
- The code computes `total_samples = sum(client_lens.values())` (line 153) and `weight = client_lens[client_id] / total_samples` (line 159) then does per-key accumulation: `global_dict[k] += state[k] * weight` (line 161).
- This produces an exact per-parameter weighted average across clients.

### 2) Data splitting

IID split (`iid_split`, lines 42-52):
- Randomly shuffle dataset indices (line 46).
- Divide into `num_clients` contiguous chunks of size floor(len(dataset)/num_clients) (line 44 and 47).
- Any remainder indices are appended to the last client (lines 48-51).

Non-IID split (`non_iid_split`, lines 55-78):
- This follows the common shard-based non-iid partitioning:
  1. Sort all dataset indices by label (lines 57-62).
  2. Partition sorted indices into `num_shards` equal shards (line 63-64).
  3. Shuffle the shards and assign `shards_per_client` random shards to each client (lines 65-72).
  4. Any remaining shards are distributed round-robin (lines 73-77).
- The effect: each client tends to receive examples concentrated in a small subset of labels.

### 3) Local training on clients (`train_local`, lines 119-132)

- Each client trains a local copy of the model (copy of global parameters) for `epochs` epochs using SGD (default lr=0.01).
- Loss: cross entropy (classification) — `nn.CrossEntropyLoss()` (line 122).
- Optimizer update loop: zero_grad, forward, compute loss, backward, optimizer.step (lines 124-131).
- The function returns the client's state dict after local training (line 132).

Notes / possible improvements:
- Optimizer choice is fixed to SGD in this version. Using Adam or different lrs often improves convergence for small/quick runs.
- There is no local validation or early stopping; clients always return final weights after requested epochs.

### 4) Communication rounds and client sampling (main loop, lines 195-221)

- For each round r in 1..R (line 195), the server optionally samples a fraction `frac` of clients (lines 197-201). By default `frac=1.0` (all clients).
- The server broadcasts the global model (line 204) by copying `global_model.state_dict()` and loading it into each client's local model (line 210).
- Each sampled client trains locally and returns its parameters (lines 208-213).
- Server aggregates returned client parameters using FedAvg (line 216).
- The global model is evaluated on the (central) test set after aggregation (lines 218-221).

### 5) Evaluation and outputs

- Evaluation measure: standard classification accuracy on `test_loader` via `evaluate()` (lines 135-147). Returns fraction correct.
- After each round the global test accuracy is appended to `acc_history` (line 220).
- After the run, the script writes:
  - `output/accuracy.png` : a plain plot of accuracy vs round (line 236)
  - `output/accuracy_annotated.png` : same plot with numeric annotations (line 242)
  - `output/accuracy.txt` : CSV file with `round,accuracy_percent` rows (lines 244-249)

### 6) Reproducibility notes

- Seeds: `random.seed`, `np.random.seed`, and `torch.manual_seed` are set from `--seed` (lines 181-183).
- Device selection: the script uses CUDA if available, otherwise CPU (line 185).
- Quick mode: when invoked with `--quick` (line 174) the script uses small subsets (2k train / 1k test) to make smoke tests fast (lines 86-99). Results on `--quick` may not reflect full-dataset behavior.

### 7) Where bugs can happen (and how we fixed one)

- Aggregation weighting: a prior bug (summing `client_lens` instead of `client_lens.values()`) caused incorrect total sample counts and invalid weights. That has been fixed in the current file (line 153).
- Device/dtype mismatches during aggregation: ensure all tensors are on the same device and have compatible dtypes; `torch.zeros_like` is used to initialize accumulators (line 157), which creates tensors matching `global_dict[k]` dtype/device.

---

## Quick reproduction commands

From repository root, after creating and activating the virtualenv used earlier, install dependencies and run:

```bash
pip install -r requirements.txt
# Quick smoke test (2 clients, 20 rounds):
/home/amir/GitHub/federated-learning-simulator/.venv/bin/python src/fedavg_simulator.py --clients 2 --rounds 20 --local-epochs 1 --quick
```

To run a more realistic experiment (full MNIST, more local epochs):

```bash
/home/amir/GitHub/federated-learning-simulator/.venv/bin/python src/fedavg_simulator.py --clients 4 --rounds 50 --local-epochs 5
```

---

If you'd like, I can also:
- Add a small unit test asserting weighted-avg properties (sum of weights equals 1, aggregated tensors close to manual computation).
- Add CLI flags to choose optimizer (`adam` vs `sgd`) and learning rate.
- Produce a side-by-side IID vs Non-IID comparison plot and CSV output.

Tell me which of the extras you want me to add and I will implement and run them.

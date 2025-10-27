import argparse
import copy
import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


# Simple CNN for MNIST
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def iid_split(dataset, num_clients):
    # Even split of indices
    num_items = len(dataset) // num_clients
    all_idxs = list(range(len(dataset)))
    random.shuffle(all_idxs)
    dict_clients = {i: all_idxs[i * num_items: (i + 1) * num_items] for i in range(num_clients)}
    # give remainder to last client
    rest = all_idxs[num_clients * num_items:]
    if rest:
        dict_clients[num_clients - 1].extend(rest)
    return dict_clients


def non_iid_split(dataset, num_clients, num_shards=200, shards_per_client=2):
    # Following a common non-iid split from literature: sort by label and divide into shards
    targets = np.array(dataset.targets)
    idxs = np.arange(len(targets))
    # sort by label
    idxs_labels = np.vstack((idxs, targets)).T
    idxs_labels = idxs_labels[idxs_labels[:, 1].argsort()]
    idxs = idxs_labels[:, 0].astype(int)
    shard_size = len(dataset) // num_shards
    shards = [list(idxs[i * shard_size: (i + 1) * shard_size]) for i in range(num_shards)]
    random.shuffle(shards)
    dict_clients = {i: [] for i in range(num_clients)}
    for i in range(num_clients):
        assigned = []
        for _ in range(shards_per_client):
            if shards:
                assigned.extend(shards.pop())
        dict_clients[i] = assigned
    # If shards remain, distribute
    si = 0
    while shards:
        dict_clients[si % num_clients].extend(shards.pop())
        si += 1
    return dict_clients


def get_dataloaders(num_clients=4, batch_size=32, quick=False, non_iid=False):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if quick:
        # Use only a fraction for smoke tests
        train_subset = list(range(2000))
        test_subset = list(range(1000))
        trainset = Subset(trainset, train_subset)
        testset = Subset(testset, test_subset)
        # For Subset, targets attribute is not available; hack: wrap
        class QuickDataset(Subset):
            @property
            def targets(self):
                return [trainset.dataset.targets[i] for i in self.indices]

        trainset = QuickDataset(trainset.dataset, trainset.indices)
        testset = Subset(testset.dataset, testset.indices)

    if non_iid:
        client_idxs = non_iid_split(trainset if not quick else trainset, num_clients)
    else:
        client_idxs = iid_split(trainset, num_clients)

    client_loaders = {}
    client_data_lens = {}
    for i in range(num_clients):
        idxs = client_idxs[i]
        subset = Subset(trainset, idxs)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders[i] = loader
        client_data_lens[i] = len(subset)

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return client_loaders, test_loader, client_data_lens


def train_local(model, dataloader, epochs, lr=0.01, device='cpu'):
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for e in range(epochs):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model.state_dict()


def evaluate(model, dataloader, device='cpu'):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return correct / total if total > 0 else 0.0


def fedavg_aggregate(global_model, client_states, client_lens):
    # Weighted average by number of samples per client
    # client_lens is a dict mapping client_id -> num_samples
    total_samples = sum(client_lens.values())
    global_dict = global_model.state_dict()
    # initialize
    for k in global_dict.keys():
        global_dict[k] = torch.zeros_like(global_dict[k])
    for client_id, state in client_states.items():
        weight = client_lens[client_id] / total_samples
        for k in global_dict.keys():
            global_dict[k] += state[k] * weight
    global_model.load_state_dict(global_dict)
    return global_model


def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Simulator (FedAvg)')
    parser.add_argument('--clients', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--frac', type=float, default=1.0, help='fraction of clients sampled each round')
    parser.add_argument('--non-iid', dest='non_iid', action='store_true')
    parser.add_argument('--quick', action='store_true', help='use small subset for quick smoke test')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    client_loaders, test_loader, client_lens = get_dataloaders(num_clients=args.clients, batch_size=args.batch_size, quick=args.quick, non_iid=args.non_iid)

    # Initialize global model
    global_model = SimpleCNN()

    acc_history = []

    for r in range(args.rounds):
        print(f"\n--- Round {r+1}/{args.rounds} ---")
        sampled_clients = list(range(args.clients))
        if args.frac < 1.0:
            sample_size = max(1, int(args.frac * args.clients))
            sampled_clients = random.sample(range(args.clients), sample_size)
        print(f"Sampled clients this round: {sampled_clients}")

        # broadcast
        global_state = copy.deepcopy(global_model.state_dict())

        client_states = {}
        # each client trains locally
        for cid in sampled_clients:
            local_model = SimpleCNN()
            local_model.load_state_dict(global_state)
            print(f"Client {cid}: training on {client_lens[cid]} samples")
            state = train_local(local_model, client_loaders[cid], epochs=args.local_epochs, device=device)
            client_states[cid] = state

        # aggregate
        global_model = fedavg_aggregate(global_model, client_states, client_lens)

        # evaluate
        acc = evaluate(global_model, test_loader, device=device)
        acc_history.append(acc)
        print(f"Global model accuracy after round {r+1}: {acc*100:.2f}%")

    # Save plot and accuracy data
    os.makedirs('output', exist_ok=True)
    rounds = list(range(1, len(acc_history) + 1))
    acc_percent = [a * 100 for a in acc_history]

    plt.figure()
    plt.plot(rounds, acc_percent, marker='o')
    plt.xlabel('Round')
    plt.ylabel('Test Accuracy (%)')
    plt.title('FedAvg: Global Model Accuracy per Round')
    plt.grid(True)

    # Save a basic plot
    plt.savefig('output/accuracy.png')

    # Annotate points and save a higher-resolution copy
    for x, y in zip(rounds, acc_percent):
        plt.annotate(f'{y:.2f}%', (x, y), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig('output/accuracy_annotated.png', dpi=200)

    # Save raw numbers to accuracy.txt (round, accuracy_percent)
    txt_path = os.path.join('output', 'accuracy.txt')
    with open(txt_path, 'w') as f:
        f.write('round,accuracy_percent\n')
        for x, y in zip(rounds, acc_percent):
            f.write(f"{x},{y:.4f}\n")

    print('\nSaved accuracy plot to output/accuracy.png')
    print('Saved annotated plot to output/accuracy_annotated.png')
    print(f'Saved accuracy data to {txt_path}')


if __name__ == '__main__':
    main()


import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# Utility to load datasets (MNIST or CIFAR-10)
def load_dataset(dataset_name, data_dir, batch_size=32, train=True, indices=None):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = datasets.MNIST(data_dir, train=train, download=True, transform=transform)
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        dataset = datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if indices is not None:
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# Utility to split dataset indices for federated clients (kept for backward compatibility)
def split_indices(num_samples, num_clients):
    indices = np.random.permutation(num_samples)
    return np.array_split(indices, num_clients)

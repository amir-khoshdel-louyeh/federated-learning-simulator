import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Utility to load MNIST dataset

def load_mnist(data_dir, batch_size=32, train=True, indices=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(data_dir, train=train, download=True, transform=transform)
    if indices is not None:
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# Utility to split dataset indices for federated clients

def split_indices(num_samples, num_clients):
    indices = np.random.permutation(num_samples)
    return np.array_split(indices, num_clients)

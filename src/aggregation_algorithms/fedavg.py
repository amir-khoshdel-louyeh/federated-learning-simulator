
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data import load_dataset, split_indices_iid, split_indices_non_iid
from models import SimpleMLP
from utils import get_weights, set_weights, average_weights

def train_fedavg(num_clients=5, num_rounds=3, local_epochs=1, batch_size=32, data_dir="../data/MNIST", partition="IID", dataset_name="MNIST"):
	# Force CPU usage
	device = torch.device("cpu")
	# Prepare data
	full_train = load_dataset(dataset_name, data_dir, batch_size=batch_size, train=True).dataset
	# Get labels for non-IID
	labels = None
	if partition == "non-IID":
		labels = [full_train[i][1] for i in range(len(full_train))]
		indices_split = split_indices_non_iid(len(full_train), num_clients, labels=labels)
	else:
		indices_split = split_indices_iid(len(full_train), num_clients)
	input_dim = 784 if dataset_name == "MNIST" else 3072
	global_model = SimpleMLP(input_dim=input_dim).to(device)
	for rnd in range(num_rounds):
		local_weights = []
		for c in range(num_clients):
			client_model = SimpleMLP(input_dim=input_dim).to(device)
			set_weights(client_model, get_weights(global_model))
			optimizer = optim.SGD(client_model.parameters(), lr=0.01)
			loader = load_dataset(dataset_name, data_dir, batch_size=batch_size, train=True, indices=indices_split[c])
			client_model.train()
			for _ in range(local_epochs):
				for x, y in loader:
					x, y = x.to(device), y.to(device)
					optimizer.zero_grad()
					out = client_model(x)
					loss = nn.CrossEntropyLoss()(out, y)
					loss.backward()
					optimizer.step()
			local_weights.append(get_weights(client_model))
		avg_weights = average_weights(local_weights)
		set_weights(global_model, avg_weights)
		print(f"Round {rnd+1} complete.")
		round_acc = evaluate(global_model, data_dir=data_dir, batch_size=batch_size, dataset_name=dataset_name, verbose=False)
		print(f"accuracy: {round_acc:.4f}")
		print(f"result: {round_acc:.4f}\n")
	return global_model


def _compute_accuracy(model, data_dir, batch_size, dataset_name):
	# Force CPU usage
	device = torch.device("cpu")
	model = model.to(device)
	loader = load_dataset(dataset_name, data_dir, batch_size=batch_size, train=False)
	model.eval()
	correct, total = 0, 0
	with torch.no_grad():
		for x, y in loader:
			x, y = x.to(device), y.to(device)
			out = model(x)
			pred = out.argmax(dim=1)
			correct += (pred == y).sum().item()
			total += y.size(0)
	return correct / total


def evaluate(model, data_dir="../data/MNIST", batch_size=32, dataset_name="MNIST", verbose=True):
	acc = _compute_accuracy(model, data_dir, batch_size, dataset_name)
	if verbose:
		print(f"Test Accuracy: {acc:.4f}")
	return acc

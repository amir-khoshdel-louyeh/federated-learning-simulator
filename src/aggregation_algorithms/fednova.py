
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data import load_dataset, split_indices_iid, split_indices_non_iid
from models import SimpleMLP
from utils import get_weights, set_weights

def train_fednova(num_clients=5, num_rounds=3, local_epochs=1, batch_size=32, data_dir="../data/MNIST", partition="IID", dataset_name="MNIST"):
	"""
	FedNova: Normalizes client updates by their effective training steps to prevent
	clients with more data from dominating the global model.
	"""
	# Force CPU usage
	device = torch.device("cpu")
	# Prepare data
	full_train = load_dataset(dataset_name, data_dir, batch_size=batch_size, train=True).dataset
	labels = None
	if partition == "non-IID":
		labels = [full_train[i][1] for i in range(len(full_train))]
		indices_split = split_indices_non_iid(len(full_train), num_clients, labels=labels)
	else:
		indices_split = split_indices_iid(len(full_train), num_clients)
	
	input_dim = 784 if dataset_name == "MNIST" else 3072
	global_model = SimpleMLP(input_dim=input_dim).to(device)
	
	for rnd in range(num_rounds):
		local_deltas = []
		local_normalizers = []
		
		for c in range(num_clients):
			client_model = SimpleMLP(input_dim=input_dim).to(device)
			global_weights = get_weights(global_model)
			set_weights(client_model, global_weights)
			optimizer = optim.SGD(client_model.parameters(), lr=0.01)
			loader = load_dataset(dataset_name, data_dir, batch_size=batch_size, train=True, indices=indices_split[c])
			
			client_model.train()
			step_count = 0
			for _ in range(local_epochs):
				for x, y in loader:
					x, y = x.to(device), y.to(device)
					optimizer.zero_grad()
					out = client_model(x)
					loss = nn.CrossEntropyLoss()(out, y)
					loss.backward()
					optimizer.step()
					step_count += 1
			
			# Compute delta (difference from global model)
			client_weights = get_weights(client_model)
			delta = {}
			for key in global_weights.keys():
				delta[key] = client_weights[key] - global_weights[key]
			
			local_deltas.append(delta)
			# Normalizer is the number of steps taken
			local_normalizers.append(step_count)
		
		# Aggregate with normalization: weighted average by inverse of steps
		total_normalizer = sum(local_normalizers)
		aggregated_delta = {}
		
		for key in global_weights.keys():
			aggregated_delta[key] = torch.zeros_like(global_weights[key])
			for c in range(num_clients):
				# Weight by effective contribution (inversely proportional to steps)
				weight = local_normalizers[c] / total_normalizer
				aggregated_delta[key] += weight * local_deltas[c][key]
		
		# Update global model
		new_weights = {}
		for key in global_weights.keys():
			new_weights[key] = global_weights[key] + aggregated_delta[key]
		
		set_weights(global_model, new_weights)
		print(f"FedNova Round {rnd+1} complete. (normalized by step counts: {local_normalizers})")
	
	return global_model

def evaluate(model, data_dir="../data/MNIST", batch_size=32, dataset_name="MNIST"):
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
	acc = correct / total
	print(f"Test Accuracy: {acc:.4f}")
	return acc

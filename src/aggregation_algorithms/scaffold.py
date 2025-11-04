
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

def train_scaffold(num_clients=5, num_rounds=3, local_epochs=1, batch_size=32, data_dir="../data/MNIST", partition="IID", dataset_name="MNIST"):
	"""
	SCAFFOLD: Uses control variates to correct client drift.
	Each client maintains a control variate that tracks the direction of its updates,
	and the server maintains a global control variate.
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
	
	# Initialize control variates (one global, one per client)
	global_weights = get_weights(global_model)
	c_global = {k: torch.zeros_like(v) for k, v in global_weights.items()}
	c_clients = [{k: torch.zeros_like(v) for k, v in global_weights.items()} for _ in range(num_clients)]
	
	for rnd in range(num_rounds):
		local_deltas = []
		new_c_clients = []
		
		for c in range(num_clients):
			client_model = SimpleMLP(input_dim=input_dim).to(device)
			global_weights = get_weights(global_model)
			set_weights(client_model, global_weights)
			
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
					
					# Apply control variate correction to gradients
					with torch.no_grad():
						for name, param in client_model.named_parameters():
							if param.grad is not None:
								# Correct gradient using control variates
								param.grad += c_global[name] - c_clients[c][name]
					
					optimizer.step()
			
			# Compute delta and new client control variate
			client_weights = get_weights(client_model)
			delta = {}
			new_c_i = {}
			for key in global_weights.keys():
				delta[key] = client_weights[key] - global_weights[key]
				# Update client control variate: c_i^{new} = c_i^{old} - c_global + delta / (lr * steps)
				# Simplified version: just use the delta direction
				new_c_i[key] = c_clients[c][key] - c_global[key] + delta[key]
			
			local_deltas.append(delta)
			new_c_clients.append(new_c_i)
		
		# Aggregate deltas (simple average)
		aggregated_delta = {}
		for key in global_weights.keys():
			aggregated_delta[key] = torch.zeros_like(global_weights[key])
			for c in range(num_clients):
				aggregated_delta[key] += local_deltas[c][key]
			aggregated_delta[key] /= num_clients
		
		# Update global model
		new_weights = {}
		for key in global_weights.keys():
			new_weights[key] = global_weights[key] + aggregated_delta[key]
		set_weights(global_model, new_weights)
		
		# Update global control variate
		for key in c_global.keys():
			delta_c = torch.zeros_like(c_global[key])
			for c in range(num_clients):
				delta_c += (new_c_clients[c][key] - c_clients[c][key])
			c_global[key] += delta_c / num_clients
		
		# Update client control variates
		c_clients = new_c_clients
		
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

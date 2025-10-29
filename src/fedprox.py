
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import load_mnist, split_indices

class SimpleMLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(28*28, 128)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(128, 10)
	def forward(self, x):
		x = self.flatten(x)
		x = self.relu(self.fc1(x))
		x = self.fc2(x)
		return x

def average_weights(w_list):
	avg = w_list[0].copy()
	for key in avg.keys():
		for i in range(1, len(w_list)):
			avg[key] += w_list[i][key]
		avg[key] = avg[key] / len(w_list)
	return avg

def set_weights(model, weights):
	model.load_state_dict(weights)

def get_weights(model):
	return {k: v.clone() for k, v in model.state_dict().items()}

def fedprox_loss(output, target, model, global_weights, mu=0.01):
	ce_loss = nn.CrossEntropyLoss()(output, target)
	prox_reg = 0.0
	for k, v in model.state_dict().items():
		prox_reg += ((v - global_weights[k]) ** 2).sum()
	return ce_loss + mu * prox_reg

def train_fedprox(num_clients=5, num_rounds=3, local_epochs=1, batch_size=32, mu=0.01, data_dir="../data/MNIST"):
	full_train = load_mnist(data_dir, batch_size=batch_size, train=True).dataset
	indices_split = split_indices(len(full_train), num_clients)
	global_model = SimpleMLP()
	for rnd in range(num_rounds):
		local_weights = []
		global_w = get_weights(global_model)
		for c in range(num_clients):
			client_model = SimpleMLP()
			set_weights(client_model, global_w)
			optimizer = optim.SGD(client_model.parameters(), lr=0.01)
			loader = load_mnist(data_dir, batch_size=batch_size, train=True, indices=indices_split[c])
			client_model.train()
			for _ in range(local_epochs):
				for x, y in loader:
					optimizer.zero_grad()
					out = client_model(x)
					loss = fedprox_loss(out, y, client_model, global_w, mu)
					loss.backward()
					optimizer.step()
			local_weights.append(get_weights(client_model))
		avg_weights = average_weights(local_weights)
		set_weights(global_model, avg_weights)
		print(f"FedProx Round {rnd+1} complete.")
	return global_model

def evaluate(model, data_dir="../data/MNIST", batch_size=32):
	loader = load_mnist(data_dir, batch_size=batch_size, train=False)
	model.eval()
	correct, total = 0, 0
	with torch.no_grad():
		for x, y in loader:
			out = model(x)
			pred = out.argmax(dim=1)
			correct += (pred == y).sum().item()
			total += y.size(0)
	acc = correct / total
	print(f"Test Accuracy: {acc:.4f}")
	return acc


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import load_mnist, split_indices

class FedPerMLP(nn.Module):
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.base = nn.Linear(28*28, 128)
		self.relu = nn.ReLU()
		self.head = nn.Linear(128, 10)
	def forward(self, x):
		x = self.flatten(x)
		x = self.relu(self.base(x))
		x = self.head(x)
		return x

def get_base_weights(model):
	return {k: v.clone() for k, v in model.state_dict().items() if 'base' in k}

def set_base_weights(model, base_weights):
	state = model.state_dict()
	for k in base_weights:
		state[k] = base_weights[k]
	model.load_state_dict(state)

def train_fedper(num_clients=5, num_rounds=3, local_epochs=1, batch_size=32, data_dir="../data/MNIST"):
	full_train = load_mnist(data_dir, batch_size=batch_size, train=True).dataset
	indices_split = split_indices(len(full_train), num_clients)
	global_base = None
	client_heads = []
	for c in range(num_clients):
		client_heads.append(None)
	for rnd in range(num_rounds):
		base_weights_list = []
		for c in range(num_clients):
			model = FedPerMLP()
			if global_base is not None:
				set_base_weights(model, global_base)
			if client_heads[c] is not None:
				state = model.state_dict()
				for k in client_heads[c]:
					state[k] = client_heads[c][k]
				model.load_state_dict(state)
			optimizer = optim.SGD(model.parameters(), lr=0.01)
			loader = load_mnist(data_dir, batch_size=batch_size, train=True, indices=indices_split[c])
			model.train()
			for _ in range(local_epochs):
				for x, y in loader:
					optimizer.zero_grad()
					out = model(x)
					loss = nn.CrossEntropyLoss()(out, y)
					loss.backward()
					optimizer.step()
			base_weights_list.append(get_base_weights(model))
			# Save personalized head
			client_heads[c] = {k: v.clone() for k, v in model.state_dict().items() if 'head' in k}
		# Aggregate base layers
		global_base = base_weights_list[0].copy()
		for k in global_base:
			for i in range(1, len(base_weights_list)):
				global_base[k] += base_weights_list[i][k]
			global_base[k] = global_base[k] / len(base_weights_list)
		print(f"FedPer Round {rnd+1} complete.")
	# Return one personalized model per client
	models = []
	for c in range(num_clients):
		model = FedPerMLP()
		set_base_weights(model, global_base)
		state = model.state_dict()
		for k in client_heads[c]:
			state[k] = client_heads[c][k]
		model.load_state_dict(state)
		models.append(model)
	return models

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

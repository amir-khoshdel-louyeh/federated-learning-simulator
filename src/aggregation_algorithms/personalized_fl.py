
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

def train_personalized_fl(num_clients=5, num_rounds=3, local_epochs=1, batch_size=32, data_dir="../data/MNIST", partition="IID", dataset_name="MNIST", finetune_epochs=2):
	"""
	Personalized FL: Trains a global model, then each client fine-tunes it
	on their local data to create a personalized model.
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
	
	# Phase 1: Standard federated training to create a good global model
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
		
		# Average weights
		avg_weights = local_weights[0].copy()
		for key in avg_weights.keys():
			for i in range(1, len(local_weights)):
				avg_weights[key] += local_weights[i][key]
			avg_weights[key] = avg_weights[key] / len(local_weights)
		
		set_weights(global_model, avg_weights)
		print(f"Round {rnd+1} complete.")
		print("(global training phase)")
		round_acc = _compute_global_accuracy(global_model, data_dir, batch_size, dataset_name)
		print(f"accuracy: {round_acc:.4f}")
		print(f"result: {round_acc:.4f}\n")
	
	# Phase 2: Personalization - each client fine-tunes the global model
	print(f"Starting personalization phase (fine-tuning for {finetune_epochs} epochs per client)...")
	personalized_models = []
	
	for c in range(num_clients):
		personal_model = SimpleMLP(input_dim=input_dim).to(device)
		set_weights(personal_model, get_weights(global_model))
		optimizer = optim.SGD(personal_model.parameters(), lr=0.005)  # Lower learning rate for fine-tuning
		loader = load_dataset(dataset_name, data_dir, batch_size=batch_size, train=True, indices=indices_split[c])
		
		personal_model.train()
		for _ in range(finetune_epochs):
			for x, y in loader:
				x, y = x.to(device), y.to(device)
				optimizer.zero_grad()
				out = personal_model(x)
				loss = nn.CrossEntropyLoss()(out, y)
				loss.backward()
				optimizer.step()
		
		personalized_models.append(personal_model)
		print(f"Client {c} personalization complete.")
	
	return personalized_models


def _compute_global_accuracy(model, data_dir, batch_size, dataset_name):
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
	return correct / total if total > 0 else 0.0


def _evaluate_personalized_models(models, data_dir, batch_size, dataset_name):
	device = torch.device("cpu")
	loader = load_dataset(dataset_name, data_dir, batch_size=batch_size, train=False)
	accuracies = []
	for model in models:
		model = model.to(device)
		model.eval()
		correct, total = 0, 0
		with torch.no_grad():
			for x, y in loader:
				x, y = x.to(device), y.to(device)
				out = model(x)
				pred = out.argmax(dim=1)
				correct += (pred == y).sum().item()
				total += y.size(0)
		accuracies.append(correct / total if total > 0 else 0.0)
	return accuracies


def evaluate(models, data_dir="../data/MNIST", batch_size=32, dataset_name="MNIST", verbose=True):
	"""
	Evaluate all personalized models and return their average accuracy.
	"""
	accuracies = _evaluate_personalized_models(models, data_dir, batch_size, dataset_name)
	if verbose:
		for idx, acc in enumerate(accuracies):
			print(f"Client {idx} personalized model accuracy: {acc:.4f}")
		avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
		print(f"Average personalized accuracy: {avg_acc:.4f}")
	return accuracies  # Return list of accuracies

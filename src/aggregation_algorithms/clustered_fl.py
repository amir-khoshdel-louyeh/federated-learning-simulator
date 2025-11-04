
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
from sklearn.cluster import KMeans

def train_clustered_fl(num_clients=5, num_rounds=3, local_epochs=1, batch_size=32, data_dir="../data/MNIST", partition="IID", dataset_name="MNIST", num_clusters=2):
	"""
	Clustered FL: Groups clients with similar data distributions and builds
	a separate global model for each cluster.
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
	
	# Initialize one model per cluster
	cluster_models = [SimpleMLP(input_dim=input_dim).to(device) for _ in range(num_clusters)]
	
	# Assign clients to clusters (simple: based on client ID modulo for now, or use label distribution)
	# For a more realistic clustering, we compute label distribution and use KMeans
	client_label_distributions = []
	for c in range(num_clients):
		loader = load_dataset(dataset_name, data_dir, batch_size=batch_size, train=True, indices=indices_split[c])
		label_counts = torch.zeros(10)
		for _, y in loader:
			for label in y:
				label_counts[label.item()] += 1
		# Normalize to get distribution
		label_counts = label_counts / label_counts.sum()
		client_label_distributions.append(label_counts.numpy())
	
	# Cluster clients based on label distribution
	client_label_distributions = np.array(client_label_distributions)
	kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
	cluster_assignments = kmeans.fit_predict(client_label_distributions)
	
	print(f"Client cluster assignments: {cluster_assignments}")
	
	for rnd in range(num_rounds):
		# Train each cluster separately
		for cluster_id in range(num_clusters):
			clients_in_cluster = [c for c in range(num_clients) if cluster_assignments[c] == cluster_id]
			
			if len(clients_in_cluster) == 0:
				continue
			
			local_weights = []
			for c in clients_in_cluster:
				client_model = SimpleMLP(input_dim=input_dim).to(device)
				set_weights(client_model, get_weights(cluster_models[cluster_id]))
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
			
			# Average weights within the cluster
			avg_weights = average_weights(local_weights)
			set_weights(cluster_models[cluster_id], avg_weights)
		
		print(f"Round {rnd+1} complete.")
		round_acc = evaluate(cluster_models, data_dir=data_dir, batch_size=batch_size, dataset_name=dataset_name, verbose=False)
		print(f"accuracy: {round_acc:.4f}")
		print(f"result: {round_acc:.4f}\n")
	
	# Return all cluster models (we'll evaluate the best one or average them)
	return cluster_models


def _evaluate_clusters(models, data_dir, batch_size, dataset_name):
	"""
	Compute accuracies for all cluster models and return per-cluster and best accuracies.
	"""
	# Force CPU usage
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
	best_acc = max(accuracies) if accuracies else 0.0
	return accuracies, best_acc


def evaluate(models, data_dir="../data/MNIST", batch_size=32, dataset_name="MNIST", verbose=True):
	accuracies, best_acc = _evaluate_clusters(models, data_dir, batch_size, dataset_name)
	if verbose:
		for idx, acc in enumerate(accuracies):
			print(f"Cluster {idx} Test Accuracy: {acc:.4f}")
		print(f"Best Cluster Accuracy: {best_acc:.4f}")
	return best_acc


import torch
import torch.nn as nn
import torch.optim as optim
from utils import load_mnist

class ClientNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.fc = nn.Linear(28*28, 128)
		self.relu = nn.ReLU()
	def forward(self, x):
		x = self.flatten(x)
		x = self.relu(self.fc(x))
		return x

class ServerNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc = nn.Linear(128, 10)
	def forward(self, x):
		x = self.fc(x)
		return x

def train_split_learning(num_epochs=3, batch_size=32, data_dir="../data/MNIST"):
	client = ClientNet()
	server = ServerNet()
	optimizer_client = optim.SGD(client.parameters(), lr=0.01)
	optimizer_server = optim.SGD(server.parameters(), lr=0.01)
	loader = load_mnist(data_dir, batch_size=batch_size, train=True)
	for epoch in range(num_epochs):
		for x, y in loader:
			# Forward on client
			feat = client(x)
			# Forward on server
			out = server(feat)
			loss = nn.CrossEntropyLoss()(out, y)
			# Backward on server
			optimizer_server.zero_grad()
			optimizer_client.zero_grad()
			loss.backward()
			# Send gradients to client (autograd handles this)
			optimizer_server.step()
			optimizer_client.step()
		print(f"Split Learning Epoch {epoch+1} complete.")
	return client, server

def evaluate(client, server, data_dir="../data/MNIST", batch_size=32):
	loader = load_mnist(data_dir, batch_size=batch_size, train=False)
	client.eval()
	server.eval()
	correct, total = 0, 0
	with torch.no_grad():
		for x, y in loader:
			feat = client(x)
			out = server(feat)
			pred = out.argmax(dim=1)
			correct += (pred == y).sum().item()
			total += y.size(0)
	acc = correct / total
	print(f"Test Accuracy: {acc:.4f}")
	return acc

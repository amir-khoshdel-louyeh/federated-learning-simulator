"""
Neural network models for federated learning.
"""
import torch.nn as nn


class SimpleMLP(nn.Module):
	"""
	Simple Multi-Layer Perceptron for MNIST and CIFAR-10.
	
	Args:
		input_dim (int): Input dimension (784 for MNIST, 3072 for CIFAR-10)
	"""
	def __init__(self, input_dim=784):
		super().__init__()
		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(input_dim, 128)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(128, 10)
	
	def forward(self, x):
		x = self.flatten(x)
		x = self.relu(self.fc1(x))
		x = self.fc2(x)
		return x

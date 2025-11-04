"""
Common utility functions for model operations.
"""
import torch


def get_weights(model):
	"""
	Extract model weights as a dictionary.
	
	Args:
		model: PyTorch model
		
	Returns:
		dict: Cloned state dictionary
	"""
	return {k: v.clone() for k, v in model.state_dict().items()}


def set_weights(model, weights):
	"""
	Set model weights from a dictionary.
	
	Args:
		model: PyTorch model
		weights: State dictionary to load
	"""
	model.load_state_dict(weights)


def average_weights(w_list):
	"""
	Average a list of model weight dictionaries.
	
	Args:
		w_list: List of state dictionaries
		
	Returns:
		dict: Averaged state dictionary
	"""
	avg = w_list[0].copy()
	for key in avg.keys():
		for i in range(1, len(w_list)):
			avg[key] += w_list[i][key]
		avg[key] = avg[key] / len(w_list)
	return avg

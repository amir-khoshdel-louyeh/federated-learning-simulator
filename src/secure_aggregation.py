
import numpy as np

def encrypt(weights, key):
	# Mock encryption: add key
	return {k: v + key for k, v in weights.items()}

def decrypt(weights, key):
	# Mock decryption: subtract key
	return {k: v - key for k, v in weights.items()}

def secure_aggregate(weights_list, key_list):
	# Each client encrypts with its key
	encrypted = [encrypt(w, k) for w, k in zip(weights_list, key_list)]
	# Server sums encrypted weights
	agg = encrypted[0].copy()
	for k in agg:
		for i in range(1, len(encrypted)):
			agg[k] += encrypted[i][k]
	# Server decrypts sum (sum of keys)
	total_key = sum(key_list)
	agg = {k: v - total_key * len(weights_list) for k, v in agg.items()}
	# Average
	agg = {k: v / len(weights_list) for k, v in agg.items()}
	return agg

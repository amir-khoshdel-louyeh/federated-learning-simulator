import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import sys
import os
import io


sys.path.append(os.path.dirname(__file__))

from aggregation_algorithms import fedavg, fedopt, fedprox, fednova, scaffold, clustered_fl, personalized_fl
from data import split_indices_iid, split_indices_non_iid
from evaluation import print_sorted_results, format_sorted_results


class TextRedirector(io.StringIO):
	"""Redirect text writes to a Tkinter widget in a thread-safe way."""

	def __init__(self, widget):
		super().__init__()
		self.widget = widget

	def write(self, message):
		if not message:
			return
		def append():
			self.widget.insert(tk.END, message)
			self.widget.see(tk.END)
		self.widget.after(0, append)

	def flush(self):
		pass

def get_algo_func(algo, params):
	partition = params.get('partition', 'IID')
	dataset_name = params.get('dataset_name', 'MNIST')
	if algo == "FedAvg":
		return lambda: fedavg.train_fedavg(num_clients=params['clients'], num_rounds=params['rounds'], local_epochs=params['local_epochs'], batch_size=params['batch_size'], partition=partition, dataset_name=dataset_name)
	if algo == "FedOpt":
		return lambda: fedopt.train_fedopt(num_clients=params['clients'], num_rounds=params['rounds'], local_epochs=params['local_epochs'], batch_size=params['batch_size'], partition=partition, dataset_name=dataset_name)
	if algo == "FedProx":
		return lambda: fedprox.train_fedprox(num_clients=params['clients'], num_rounds=params['rounds'], local_epochs=params['local_epochs'], batch_size=params['batch_size'], partition=partition, dataset_name=dataset_name)
	if algo == "FedNova":
		return lambda: fednova.train_fednova(num_clients=params['clients'], num_rounds=params['rounds'], local_epochs=params['local_epochs'], batch_size=params['batch_size'], partition=partition, dataset_name=dataset_name)
	if algo == "SCAFFOLD":
		return lambda: scaffold.train_scaffold(num_clients=params['clients'], num_rounds=params['rounds'], local_epochs=params['local_epochs'], batch_size=params['batch_size'], partition=partition, dataset_name=dataset_name)
	if algo == "Clustered FL":
		return lambda: clustered_fl.train_clustered_fl(num_clients=params['clients'], num_rounds=params['rounds'], local_epochs=params['local_epochs'], batch_size=params['batch_size'], partition=partition, dataset_name=dataset_name)
	if algo == "Personalized FL":
		return lambda: personalized_fl.train_personalized_fl(num_clients=params['clients'], num_rounds=params['rounds'], local_epochs=params['local_epochs'], batch_size=params['batch_size'], partition=partition, dataset_name=dataset_name)

def get_eval_func(algo):
	def get_eval_with_dataset(func):
		return lambda *args, **kwargs: func(*args, dataset_name=dataset_var.get(), **kwargs)
	if algo == "FedAvg":
		return get_eval_with_dataset(fedavg.evaluate)
	if algo == "FedOpt":
		return get_eval_with_dataset(fedopt.evaluate)
	if algo == "FedProx":
		return get_eval_with_dataset(fedprox.evaluate)
	if algo == "FedNova":
		return get_eval_with_dataset(fednova.evaluate)
	if algo == "SCAFFOLD":
		return get_eval_with_dataset(scaffold.evaluate)
	if algo == "Clustered FL":
		return lambda models: clustered_fl.evaluate(models, dataset_name=dataset_var.get())
	if algo == "Personalized FL":
		return lambda models: personalized_fl.evaluate(models, dataset_name=dataset_var.get())

def run_algorithm(algo, params, output_box):
	redirector = TextRedirector(output_box)
	old_stdout, old_stderr = sys.stdout, sys.stderr
	sys.stdout = sys.stderr = redirector
	results = None
	try:
		print(f"Running {algo} with params: {params}")
		result = get_algo_func(algo, params)()
		print("Training finished. Evaluating...")
		eval_result = get_eval_func(algo)(result)
		results = eval_result
		print(f"Result: {eval_result}\n")
	except Exception as e:
		print(f"Error: {e}")
	finally:
		sys.stdout, sys.stderr = old_stdout, old_stderr
	return results

def run_all_algorithms(params, output_box):
	algos = ["FedAvg", "FedOpt", "FedProx", "FedNova", "SCAFFOLD", "Clustered FL", "Personalized FL"]
	results = {}
	redirector = TextRedirector(output_box)
	old_stdout, old_stderr = sys.stdout, sys.stderr
	sys.stdout = sys.stderr = redirector
	try:
		separator = "=" * 20
		for idx, algo in enumerate(algos):
			if idx > 0:
				print(separator)
				print()
			print(f"--- Running {algo} ---")
			result = get_algo_func(algo, params)()
			print("Training finished. Evaluating...")
			eval_result = get_eval_func(algo)(result)
			results[algo] = eval_result
			print(f"Result: {eval_result}\n")
		print(format_sorted_results(results))
	except Exception as e:
		print(f"Error: {e}")
	finally:
		sys.stdout, sys.stderr = old_stdout, old_stderr
	if results:
		print_sorted_results(results)

def on_run():
	algo = algo_var.get()
	params = {
		'clients': int(clients_var.get()),
		'rounds': int(rounds_var.get()),
		'local_epochs': int(local_epochs_var.get()),
		'batch_size': int(batch_size_var.get()),
		'partition': partition_var.get(),
		'dataset_name': dataset_var.get(),
	}
	run_btn.config(state=tk.DISABLED)
	all_btn.config(state=tk.DISABLED)
	def task():
		run_algorithm(algo, params, output_box)
		run_btn.config(state=tk.NORMAL)
		all_btn.config(state=tk.NORMAL)
	threading.Thread(target=task).start()

def on_run_all():
	params = {
		'clients': int(clients_var.get()),
		'rounds': int(rounds_var.get()),
		'local_epochs': int(local_epochs_var.get()),
		'batch_size': int(batch_size_var.get()),
		'partition': partition_var.get(),
		'dataset_name': dataset_var.get(),
	}
	run_btn.config(state=tk.DISABLED)
	all_btn.config(state=tk.DISABLED)
	def task():
		run_all_algorithms(params, output_box)
		run_btn.config(state=tk.NORMAL)
		all_btn.config(state=tk.NORMAL)
	threading.Thread(target=task).start()



root = tk.Tk()
root.title("Federated Learning Simulator")
root.geometry("540x600")

mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.pack(fill=tk.BOTH, expand=True)

ttk.Label(mainframe, text="Select Algorithm:").pack(anchor=tk.W)
algo_var = tk.StringVar(value="FedAvg")
algo_menu = ttk.Combobox(mainframe, textvariable=algo_var, state="readonly")
algo_menu['values'] = ["FedAvg", "FedOpt", "FedProx", "FedNova", "SCAFFOLD", "Clustered FL", "Personalized FL"]
algo_menu.pack(fill=tk.X, pady=5)

# Dataset selection (after algorithm selection)
ttk.Label(mainframe, text="Select Dataset:").pack(anchor=tk.W)
dataset_var = tk.StringVar(value="MNIST")
dataset_menu = ttk.Combobox(mainframe, textvariable=dataset_var, state="readonly")
dataset_menu['values'] = ["MNIST", "CIFAR10"]
dataset_menu.pack(fill=tk.X, pady=5)

# Partitioning strategy
ttk.Label(mainframe, text="Data Partitioning:").pack(anchor=tk.W)
partition_var = tk.StringVar(value="IID")
partition_menu = ttk.Combobox(mainframe, textvariable=partition_var, state="readonly")
partition_menu['values'] = ["IID", "non-IID"]
partition_menu.pack(fill=tk.X, pady=5)

params_frame = ttk.LabelFrame(mainframe, text="Parameters")
params_frame.pack(fill=tk.X, pady=5)

clients_var = tk.StringVar(value="5")
rounds_var = tk.StringVar(value="3")
local_epochs_var = tk.StringVar(value="1")
batch_size_var = tk.StringVar(value="32")

ttk.Label(params_frame, text="Number of Clients:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
ttk.Entry(params_frame, textvariable=clients_var, width=8).grid(row=0, column=1, padx=2, pady=2)

ttk.Label(params_frame, text="Number of Rounds:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
ttk.Entry(params_frame, textvariable=rounds_var, width=8).grid(row=1, column=1, padx=2, pady=2)

ttk.Label(params_frame, text="Local Epochs:").grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
ttk.Entry(params_frame, textvariable=local_epochs_var, width=8).grid(row=2, column=1, padx=2, pady=2)

ttk.Label(params_frame, text="Batch Size:").grid(row=3, column=0, sticky=tk.W, padx=2, pady=2)
ttk.Entry(params_frame, textvariable=batch_size_var, width=8).grid(row=3, column=1, padx=2, pady=2)

for i in range(4):
	params_frame.grid_rowconfigure(i, weight=1)
	params_frame.grid_columnconfigure(i, weight=1)


run_btn = ttk.Button(mainframe, text="Run", command=on_run)
run_btn.pack(pady=5)

# Run all algorithms button
all_btn = ttk.Button(mainframe, text="Run All Algorithms", command=on_run_all)
all_btn.pack(pady=5)


output_box = scrolledtext.ScrolledText(mainframe, height=18)
output_box.pack(fill=tk.BOTH, expand=True, pady=5)


root.mainloop()

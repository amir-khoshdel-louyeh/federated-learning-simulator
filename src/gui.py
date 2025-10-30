

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import sys
import os
import io


sys.path.append(os.path.dirname(__file__))

import fedavg
import fedopt
import fedper
import fedprox

import fedsgd
import split_learning
from partition_utils import split_indices_iid, split_indices_non_iid
import result as result_module

def get_algo_func(algo, params):
	partition = params.get('partition', 'IID')
	dataset_name = params.get('dataset_name', 'MNIST')
	if algo == "FedAvg":
		return lambda: fedavg.train_fedavg(num_clients=params['clients'], num_rounds=params['rounds'], local_epochs=params['local_epochs'], batch_size=params['batch_size'], partition=partition, dataset_name=dataset_name)
	if algo == "FedOpt":
		return lambda: fedopt.train_fedopt(num_clients=params['clients'], num_rounds=params['rounds'], local_epochs=params['local_epochs'], batch_size=params['batch_size'], partition=partition, dataset_name=dataset_name)
	if algo == "FedPer":
		return lambda: fedper.train_fedper(num_clients=params['clients'], num_rounds=params['rounds'], local_epochs=params['local_epochs'], batch_size=params['batch_size'], partition=partition, dataset_name=dataset_name)
	if algo == "FedProx":
		return lambda: fedprox.train_fedprox(num_clients=params['clients'], num_rounds=params['rounds'], local_epochs=params['local_epochs'], batch_size=params['batch_size'], partition=partition, dataset_name=dataset_name)
	if algo == "FedSGD":
		return lambda: fedsgd.train_fedsgd(num_clients=params['clients'], num_rounds=params['rounds'], batch_size=params['batch_size'], partition=partition, dataset_name=dataset_name)
	if algo == "Split Learning":
		return lambda: split_learning.train_split_learning(num_epochs=params['rounds'], batch_size=params['batch_size'], dataset_name=dataset_name)

def get_eval_func(algo):
	def get_eval_with_dataset(func):
		return lambda *args, **kwargs: func(*args, dataset_name=dataset_var.get(), **kwargs)
	if algo == "FedAvg":
		return get_eval_with_dataset(fedavg.evaluate)
	if algo == "FedOpt":
		return get_eval_with_dataset(fedopt.evaluate)
	if algo == "FedPer":
		return lambda models: [fedper.evaluate(m, dataset_name=dataset_var.get()) for m in models]
	if algo == "FedProx":
		return get_eval_with_dataset(fedprox.evaluate)
	if algo == "FedSGD":
		return get_eval_with_dataset(fedsgd.evaluate)
	if algo == "Split Learning":
		return lambda tup: split_learning.evaluate(*tup, dataset_name=dataset_var.get())

def run_algorithm(algo, params, output_box):
	output_box.insert(tk.END, f"Running {algo} with params: {params}\n")
	output_box.see(tk.END)
	# Redirect stdout to capture print statements
	old_stdout = sys.stdout
	sys.stdout = mystdout = io.StringIO()
	try:
		result = get_algo_func(algo, params)()
		output_box.insert(tk.END, mystdout.getvalue())
		output_box.insert(tk.END, f"Training finished. Evaluating...\n")
		output_box.see(tk.END)
		mystdout.truncate(0)
		mystdout.seek(0)
		eval_result = get_eval_func(algo)(result)
		output_box.insert(tk.END, mystdout.getvalue())
		output_box.insert(tk.END, f"Result: {eval_result}\n\n")
		output_box.see(tk.END)
	except Exception as e:
		output_box.insert(tk.END, mystdout.getvalue())
		output_box.insert(tk.END, f"Error: {e}\n")
		output_box.see(tk.END)
	finally:
		sys.stdout = old_stdout

def run_all_algorithms(params, output_box):
	algos = ["FedAvg", "FedOpt", "FedPer", "FedProx", "FedSGD"]
	results = {}
	for algo in algos:
		output_box.insert(tk.END, f"\n--- Running {algo} ---\n")
		output_box.see(tk.END)
		old_stdout = sys.stdout
		sys.stdout = mystdout = io.StringIO()
		try:
			result = get_algo_func(algo, params)()
			output_box.insert(tk.END, mystdout.getvalue())
			output_box.insert(tk.END, f"Training finished. Evaluating...\n")
			output_box.see(tk.END)
			mystdout.truncate(0)
			mystdout.seek(0)
			eval_result = get_eval_func(algo)(result)
			output_box.insert(tk.END, mystdout.getvalue())
			output_box.insert(tk.END, f"Result: {eval_result}\n\n")
			output_box.see(tk.END)
			results[algo] = eval_result
		except Exception as e:
			output_box.insert(tk.END, mystdout.getvalue())
			output_box.insert(tk.END, f"Error: {e}\n")
			output_box.see(tk.END)
		finally:
			sys.stdout = old_stdout

	# Show sorted results in GUI output box
	def format_sorted_results(results):
		processed = {}
		for k, v in results.items():
			if isinstance(v, list):
				processed[k] = sum(v) / len(v)
			else:
				processed[k] = v
		sorted_items = sorted(processed.items(), key=lambda x: x[1], reverse=True)
		lines = ["\n===== Sorted Algorithm Results ====="]
		for i, (algo, acc) in enumerate(sorted_items, 1):
			lines.append(f"{i}. {algo}: {acc:.4f}")
		lines.append("===================================\n")
		return "\n".join(lines)

	output_box.insert(tk.END, format_sorted_results(results))
	output_box.see(tk.END)
	# Print sorted results to terminal as well
	result_module.print_sorted_results(results)

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
algo_menu['values'] = ["FedAvg", "FedOpt", "FedPer", "FedProx", "FedSGD", "Split Learning"]
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

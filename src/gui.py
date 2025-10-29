

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import sys
import os
sys.path.append(os.path.dirname(__file__))

import fedavg
import fedopt
import fedper
import fedprox
import fedsgd
import split_learning

def get_algo_func(algo, params):
	if algo == "FedAvg":
		return lambda: fedavg.train_fedavg(num_clients=params['clients'], num_rounds=params['rounds'], local_epochs=params['local_epochs'], batch_size=params['batch_size'])
	if algo == "FedOpt":
		return lambda: fedopt.train_fedopt(num_clients=params['clients'], num_rounds=params['rounds'], local_epochs=params['local_epochs'], batch_size=params['batch_size'])
	if algo == "FedPer":
		return lambda: fedper.train_fedper(num_clients=params['clients'], num_rounds=params['rounds'], local_epochs=params['local_epochs'], batch_size=params['batch_size'])
	if algo == "FedProx":
		return lambda: fedprox.train_fedprox(num_clients=params['clients'], num_rounds=params['rounds'], local_epochs=params['local_epochs'], batch_size=params['batch_size'])
	if algo == "FedSGD":
		return lambda: fedsgd.train_fedsgd(num_clients=params['clients'], num_rounds=params['rounds'], batch_size=params['batch_size'])
	if algo == "Split Learning":
		return lambda: split_learning.train_split_learning(num_epochs=params['rounds'], batch_size=params['batch_size'])

def get_eval_func(algo):
	if algo == "FedAvg":
		return lambda model: fedavg.evaluate(model)
	if algo == "FedOpt":
		return lambda model: fedopt.evaluate(model)
	if algo == "FedPer":
		return lambda models: [fedper.evaluate(m) for m in models]
	if algo == "FedProx":
		return lambda model: fedprox.evaluate(model)
	if algo == "FedSGD":
		return lambda model: fedsgd.evaluate(model)
	if algo == "Split Learning":
		return lambda tup: split_learning.evaluate(*tup)

def run_algorithm(algo, params, output_box):
	output_box.insert(tk.END, f"Running {algo} with params: {params}\n")
	output_box.see(tk.END)
	try:
		result = get_algo_func(algo, params)()
		output_box.insert(tk.END, f"Training finished. Evaluating...\n")
		output_box.see(tk.END)
		eval_result = get_eval_func(algo)(result)
		output_box.insert(tk.END, f"Result: {eval_result}\n\n")
		output_box.see(tk.END)
	except Exception as e:
		output_box.insert(tk.END, f"Error: {e}\n")
		output_box.see(tk.END)

def on_run():
	algo = algo_var.get()
	params = {
		'clients': int(clients_var.get()),
		'rounds': int(rounds_var.get()),
		'local_epochs': int(local_epochs_var.get()),
		'batch_size': int(batch_size_var.get()),
	}
	run_btn.config(state=tk.DISABLED)
	def task():
		run_algorithm(algo, params, output_box)
		run_btn.config(state=tk.NORMAL)
	threading.Thread(target=task).start()

root = tk.Tk()
root.title("Federated Learning Simulator")
root.geometry("520x520")

mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.pack(fill=tk.BOTH, expand=True)

ttk.Label(mainframe, text="Select Algorithm:").pack(anchor=tk.W)
algo_var = tk.StringVar(value="FedAvg")
algo_menu = ttk.Combobox(mainframe, textvariable=algo_var, state="readonly")
algo_menu['values'] = ["FedAvg", "FedOpt", "FedPer", "FedProx", "FedSGD", "Split Learning"]
algo_menu.pack(fill=tk.X, pady=5)

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

output_box = scrolledtext.ScrolledText(mainframe, height=18)
output_box.pack(fill=tk.BOTH, expand=True, pady=5)

root.mainloop()

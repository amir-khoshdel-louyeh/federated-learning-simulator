import os
import tkinter as tk
from tkinter import messagebox

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from model import load_mnist
from centralized import train_centralized
from federated_average import federated_average
from fedprox import fedprox
from scaffold import scaffold
from fedadam import fedadam
from fedyogi import fedyogi
from fedadagrad import fedadagrad
from fednova import fednova


def run_demo(btn, label):
    btn.config(state=tk.DISABLED)
    label.config(text="Loading and training... (this may take a minute)")
    subset = getattr(btn, "subset", 3000)
    test_size = getattr(btn, "test_size", 1000)
    clients = getattr(btn, "clients", 3)
    full_per_client = getattr(btn, "full_per_client", False)

    try:
        train_images, train_labels, test_images, test_labels = load_mnist(subset=subset, test_size=test_size)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load MNIST: {e}")
        btn.config(state=tk.NORMAL)
        return

    results = []

    # Centralized
    if getattr(btn, "run_centralized", True):
        try:
            acc = train_centralized(train_images, train_labels, test_images, test_labels)
            results.append(("Centralized", acc))
        except Exception as e:
            messagebox.showerror("Error", f"Centralized training failed: {e}")

    # FedAvg
    if getattr(btn, "run_fedavg", True):
        try:
            acc = federated_average(train_images, train_labels, test_images, test_labels, clients=clients, full_data_per_client=full_per_client)
            results.append(("FedAvg", acc))
        except Exception as e:
            messagebox.showerror("Error", f"FedAvg training failed: {e}")

    # FedProx
    if getattr(btn, "run_fedprox", False):
        try:
            acc = fedprox(train_images, train_labels, test_images, test_labels, clients=clients, full_data_per_client=full_per_client)
            results.append(("FedProx", acc))
        except Exception as e:
            messagebox.showerror("Error", f"FedProx training failed: {e}")

    # SCAFFOLD
    if getattr(btn, "run_scaffold", False):
        try:
            acc = scaffold(train_images, train_labels, test_images, test_labels, clients=clients, full_data_per_client=full_per_client)
            results.append(("SCAFFOLD", acc))
        except Exception as e:
            messagebox.showerror("Error", f"SCAFFOLD training failed: {e}")

    # FedAdam
    if getattr(btn, "run_fedadam", False):
        try:
            acc = fedadam(train_images, train_labels, test_images, test_labels, clients=clients, full_data_per_client=full_per_client)
            results.append(("FedAdam", acc))
        except Exception as e:
            messagebox.showerror("Error", f"FedAdam training failed: {e}")

    # FedYogi
    if getattr(btn, "run_fedyogi", False):
        try:
            acc = fedyogi(train_images, train_labels, test_images, test_labels, clients=clients, full_data_per_client=full_per_client)
            results.append(("FedYogi", acc))
        except Exception as e:
            messagebox.showerror("Error", f"FedYogi training failed: {e}")

    # FedAdagrad
    if getattr(btn, "run_fedadagrad", False):
        try:
            acc = fedadagrad(train_images, train_labels, test_images, test_labels, clients=clients, full_data_per_client=full_per_client)
            results.append(("FedAdagrad", acc))
        except Exception as e:
            messagebox.showerror("Error", f"FedAdagrad training failed: {e}")

    # FedNova
    if getattr(btn, "run_fednova", False):
        try:
            acc = fednova(train_images, train_labels, test_images, test_labels, clients=clients, full_data_per_client=full_per_client)
            results.append(("FedNova", acc))
        except Exception as e:
            messagebox.showerror("Error", f"FedNova training failed: {e}")

    if not results:
        label_text = "No algorithm selected."
    else:
        label_text = "    ".join(f"{name} acc: {acc:.4f}" for name, acc in results)

    label.config(text=label_text)
    btn.config(state=tk.NORMAL)


def main():
    root = tk.Tk()
    root.title("Simple MNIST: Centralized vs Federated")

    root.geometry("560x280")

    top = tk.Frame(root)
    top.pack(padx=12, pady=(10, 4), fill=tk.X)

    tk.Label(top, text="Dataset size:").grid(row=0, column=0, sticky=tk.W)
    ds_entry = tk.Entry(top, width=8)
    ds_entry.insert(0, "3000")
    ds_entry.grid(row=0, column=1, padx=(6, 18))

    tk.Label(top, text="Test size:").grid(row=0, column=2, sticky=tk.W)
    ts_entry = tk.Entry(top, width=8)
    ts_entry.insert(0, "1000")
    ts_entry.grid(row=0, column=3, padx=(6, 18))

    tk.Label(top, text="Clients:").grid(row=0, column=4, sticky=tk.W)
    clients_entry = tk.Entry(top, width=6)
    clients_entry.insert(0, "3")
    clients_entry.grid(row=0, column=5, padx=(6, 0))

    full_var = tk.BooleanVar(value=False)
    chk = tk.Checkbutton(root, text="Give full dataset to each client", variable=full_var)
    chk.pack(padx=12, pady=(6, 0), anchor=tk.W)

    # Algorithms selection
    alg_frame = tk.Frame(root)
    alg_frame.pack(padx=12, pady=(6, 0), anchor=tk.W)
    tk.Label(alg_frame, text="Algorithms:").grid(row=0, column=0, sticky=tk.W)
    central_var = tk.BooleanVar(value=True)
    fedavg_var = tk.BooleanVar(value=True)
    fedprox_var = tk.BooleanVar(value=False)
    scaffold_var = tk.BooleanVar(value=False)
    fedadam_var = tk.BooleanVar(value=False)
    fedyogi_var = tk.BooleanVar(value=False)
    fedadagrad_var = tk.BooleanVar(value=False)
    fednova_var = tk.BooleanVar(value=False)
    
    cb1 = tk.Checkbutton(alg_frame, text="Centralized", variable=central_var)
    cb2 = tk.Checkbutton(alg_frame, text="FedAvg", variable=fedavg_var)
    cb3 = tk.Checkbutton(alg_frame, text="FedProx", variable=fedprox_var)
    cb4 = tk.Checkbutton(alg_frame, text="SCAFFOLD", variable=scaffold_var)
    cb5 = tk.Checkbutton(alg_frame, text="FedAdam", variable=fedadam_var)
    cb6 = tk.Checkbutton(alg_frame, text="FedYogi", variable=fedyogi_var)
    cb7 = tk.Checkbutton(alg_frame, text="FedAdagrad", variable=fedadagrad_var)
    cb8 = tk.Checkbutton(alg_frame, text="FedNova", variable=fednova_var)
    
    cb1.grid(row=0, column=1, padx=(8, 4))
    cb2.grid(row=0, column=2, padx=(8, 4))
    cb3.grid(row=0, column=3, padx=(8, 4))
    cb4.grid(row=0, column=4, padx=(8, 4))
    cb5.grid(row=1, column=1, padx=(8, 4))
    cb6.grid(row=1, column=2, padx=(8, 4))
    cb7.grid(row=1, column=3, padx=(8, 4))
    cb8.grid(row=1, column=4, padx=(8, 4))

    btn = tk.Button(root, text="Run demo (loads MNIST and trains)", width=48)
    btn.pack(padx=12, pady=(8, 6))
    lbl = tk.Label(root, text="Ready")
    lbl.pack(padx=12, pady=(0, 12))

    def on_click():
        try:
            btn.subset = int(ds_entry.get())
        except Exception:
            btn.subset = 3000
        try:
            btn.test_size = int(ts_entry.get())
        except Exception:
            btn.test_size = 1000
        try:
            btn.clients = int(clients_entry.get())
        except Exception:
            btn.clients = 3
        btn.full_per_client = bool(full_var.get())
        # algorithm selections
        btn.run_centralized = bool(central_var.get())
        btn.run_fedavg = bool(fedavg_var.get())
        btn.run_fedprox = bool(fedprox_var.get())
        btn.run_scaffold = bool(scaffold_var.get())
        btn.run_fedadam = bool(fedadam_var.get())
        btn.run_fedyogi = bool(fedyogi_var.get())
        btn.run_fedadagrad = bool(fedadagrad_var.get())
        btn.run_fednova = bool(fednova_var.get())
        run_demo(btn, lbl)

    btn.config(command=on_click)

    root.mainloop()


if __name__ == "__main__":
    main()

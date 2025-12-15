import os
import threading
import tkinter as tk
from tkinter import messagebox
import subprocess
from tkinter import Toplevel
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from comparison import read_results, generate_figures

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


def main():
    root = tk.Tk()
    root.title("Simple MNIST: Centralized vs Federated")

    root.geometry("900x500")

    top = tk.Frame(root)
    top.pack(padx=12, pady=(10, 4), fill=tk.X)

    tk.Label(top, text="Total size:").grid(row=0, column=0, sticky=tk.W)
    ds_entry = tk.Entry(top, width=10)
    ds_entry.insert(0, "10000")
    ds_entry.grid(row=0, column=1, padx=(6, 8))
    # Inline preview of split sizes (80/10/10)
    sizes_preview = tk.Label(top, text="train: 8000, val: 1000, test: 1000")
    sizes_preview.grid(row=0, column=2, padx=(6, 0), sticky=tk.W)

    tk.Label(top, text="Clients:").grid(row=1, column=0, sticky=tk.W)
    clients_entry = tk.Entry(top, width=6)
    clients_entry.insert(0, "3")
    clients_entry.grid(row=1, column=1, padx=(6, 0))

    tk.Label(top, text="Rounds:").grid(row=1, column=2, sticky=tk.W, padx=(12, 0))
    rounds_entry = tk.Entry(top, width=6)
    rounds_entry.insert(0, "3")
    rounds_entry.grid(row=1, column=3, padx=(6, 0))

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

    # Output log area
    out_frame = tk.Frame(root)
    out_frame.pack(padx=12, pady=(8, 6), fill=tk.BOTH, expand=True)
    out_text = tk.Text(out_frame, height=8)
    out_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll = tk.Scrollbar(out_frame, command=out_text.yview)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    out_text.configure(yscrollcommand=scroll.set)

    # New Run button
    btn_run = tk.Button(root, text="Run selected algorithms", width=32)
    btn_run.pack(padx=12, pady=(0, 10))


    # Run button click handler
    def on_run_click():
        # read user inputs
        try:
            total_dataset = int(ds_entry.get())
        except Exception:
            total_dataset = 10000
        try:
            clients = int(clients_entry.get())
        except Exception:
            clients = 3
        try:
            rounds = int(rounds_entry.get())
        except Exception:
            rounds = 3
        full_per_client = bool(full_var.get())
        algs = {
            "Centralized": bool(central_var.get()),
            "FedAvg": bool(fedavg_var.get()),
            "FedProx": bool(fedprox_var.get()),
            "SCAFFOLD": bool(scaffold_var.get()),
            "FedAdam": bool(fedadam_var.get()),
            "FedYogi": bool(fedyogi_var.get()),
            "FedAdagrad": bool(fedadagrad_var.get()),
            "FedNova": bool(fednova_var.get()),
        }

        btn_run.config(state=tk.DISABLED)
        out_text.delete("1.0", tk.END)
        out_text.insert(tk.END, "Loading and training...\n")

        # storage for results to write to result.txt
        results_store = {}

        def append_log(line: str):
            out_text.insert(tk.END, line + "\n")
            out_text.see(tk.END)

        def report(algo_name, round_num, metrics):
            # accumulate metrics in results_store
            key = algo_name.lower()
            if key not in results_store:
                results_store[key] = {
                    "Accuracy": [],
                    "Convergence": [],
                    "Communication Cost": [],
                    "Stability / Variance": [],
                    "Training Time": [],
                    "Velocity": [],
                }
            results_store[key]["Accuracy"].append(metrics.get("accuracy", 0.0))
            results_store[key]["Convergence"].append(metrics.get("convergence", 0.0))
            results_store[key]["Communication Cost"].append(metrics.get("communication_cost", 0))
            results_store[key]["Stability / Variance"].append(metrics.get("stability_variance", 0.0))
            results_store[key]["Training Time"].append(metrics.get("training_time", 0.0))
            results_store[key]["Velocity"].append(metrics.get("velocity", 0.0))

            line = (
                f"{algo_name} — Round {round_num}: "
                f"acc={metrics.get('accuracy', 0):.4f}, "
                f"conv={metrics.get('convergence', 0):+.4f}, "
                f"comm={metrics.get('communication_cost', 0)}, "
                f"var={metrics.get('stability_variance', 0):.6f}, "
                f"time={metrics.get('training_time', 0):.3f}s, "
                f"vel={metrics.get('velocity', 0):.3f}"
            )
            # ensure UI update on main thread
            root.after(0, append_log, line)

        def worker():
            # Compute split sizes
            train_target = int(total_dataset * 0.8)
            test_target = int(total_dataset * 0.1)
            val_target = total_dataset - train_target - test_target
            try:
                train_images, train_labels, val_images, val_labels, test_images, test_labels = load_mnist(
                    train_size=train_target, val_size=val_target, test_size=test_target
                )
            except Exception as e:
                root.after(0, messagebox.showerror, "Error", f"Failed to load MNIST: {e}")
                root.after(0, btn_run.config, {"state": tk.NORMAL})
                return

            # Run selected algorithms
            try:
                if algs["Centralized"]:
                    # single-run metrics for centralized (no communication)
                    import time as _t
                    t0 = _t.perf_counter()
                    acc = train_centralized(train_images, train_labels, test_images, test_labels)
                    dt = _t.perf_counter() - t0
                    report("Centralized", 1, {
                        "accuracy": acc,
                        "convergence": 0.0,
                        "communication_cost": 0,
                        "stability_variance": 0.0,
                        "training_time": dt,
                        "velocity": 0.0,
                    })

                if algs["FedAvg"]:
                    federated_average(train_images, train_labels, test_images, test_labels,
                                      clients=clients, rounds=rounds, full_data_per_client=full_per_client,
                                      report=report)
                if algs["FedProx"]:
                    fedprox(train_images, train_labels, test_images, test_labels,
                            clients=clients, rounds=rounds, full_data_per_client=full_per_client,
                            report=report)
                if algs["SCAFFOLD"]:
                    scaffold(train_images, train_labels, test_images, test_labels,
                             clients=clients, rounds=rounds, full_data_per_client=full_per_client,
                             report=report)
                if algs["FedAdam"]:
                    fedadam(train_images, train_labels, test_images, test_labels,
                            clients=clients, rounds=rounds, full_data_per_client=full_per_client,
                            report=report)
                if algs["FedYogi"]:
                    fedyogi(train_images, train_labels, test_images, test_labels,
                            clients=clients, rounds=rounds, full_data_per_client=full_per_client,
                            report=report)
                if algs["FedAdagrad"]:
                    fedadagrad(train_images, train_labels, test_images, test_labels,
                               clients=clients, rounds=rounds, full_data_per_client=full_per_client,
                               report=report)
                if algs["FedNova"]:
                    fednova(train_images, train_labels, test_images, test_labels,
                            clients=clients, rounds=rounds, full_data_per_client=full_per_client,
                            report=report)
                # after all selected algorithms complete, write result.txt
                try:
                    result_path = os.path.join(os.path.dirname(__file__), "result.txt")
                    with open(result_path, "w") as f:
                        for algo_key, metrics_map in results_store.items():
                            # write as python-like dict per line: algo={...}
                            f.write(f"{algo_key}={metrics_map}\n")
                    root.after(0, append_log, f"Saved results to {result_path}")
                    # Embed comparison graphs inside the app
                    try:
                        results = read_results(result_path)
                        def show_figs():
                            # Create figures on the main thread to avoid GUI warnings/errors
                            figs = generate_figures(results)
                            if not figs:
                                append_log("No figures generated for comparison.")
                                return
                            win = Toplevel(root)
                            win.title("Algorithm Comparison")
                            win.geometry("900x700")

                            # Scrollable area setup
                            container = tk.Frame(win)
                            container.pack(fill=tk.BOTH, expand=True)

                            scroll_y = tk.Scrollbar(container, orient=tk.VERTICAL)
                            scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

                            scroll_x = tk.Scrollbar(container, orient=tk.HORIZONTAL)
                            scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

                            canvas_widget = tk.Canvas(container, yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
                            canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                            scroll_y.config(command=canvas_widget.yview)
                            scroll_x.config(command=canvas_widget.xview)

                            inner = tk.Frame(canvas_widget)
                            # add inner frame to canvas
                            canvas_widget.create_window((0, 0), window=inner, anchor="nw")

                            canvases = []
                            # Add each figure to the inner frame
                            for fig in figs:
                                fig_canvas = FigureCanvasTkAgg(fig, master=inner)
                                fig_canvas.draw()
                                widget = fig_canvas.get_tk_widget()
                                widget.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
                                canvases.append(fig_canvas)

                            # Update scroll region when inner frame changes size
                            def on_configure(event=None):
                                canvas_widget.configure(scrollregion=canvas_widget.bbox("all"))
                            inner.bind("<Configure>", on_configure)

                            append_log("Displayed comparison graphs inside the app (scroll enabled).")
                        root.after(0, show_figs)
                    except Exception as e:
                        root.after(0, append_log, f"Failed to embed comparison: {e}")
                except Exception as e:
                    root.after(0, append_log, f"Failed to save results: {e}")
            finally:
                root.after(0, btn_run.config, {"state": tk.NORMAL})

        threading.Thread(target=worker, daemon=True).start()


    # Update sizes preview when total size changes (80/10/10)
    def update_size_preview(*_):
        try:
            total = int(ds_entry.get())
        except Exception:
            total = 10000
        train = int(total * 0.8)
        test = int(total * 0.1)
        val = total - train - test
        sizes_preview.config(text=f"train: {train}, val: {val}, test: {test}")

    ds_entry.bind("<KeyRelease>", update_size_preview)
    ds_entry.bind("<FocusOut>", update_size_preview)
    update_size_preview()

    btn_run.config(command=on_run_click)

    root.mainloop()


if __name__ == "__main__":
    main()

import os
import tkinter as tk
from tkinter import messagebox

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


def load_mnist(subset=3000, test_size=1000):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0
    return train_images[:subset], train_labels[:subset], test_images[:test_size], test_labels[:test_size]


def make_model():
    inputs = tf.keras.Input(shape=(28, 28))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_centralized(x_train, y_train, x_test, y_test):
    model = make_model()
    model.fit(x_train, y_train, epochs=3, batch_size=64, verbose=0)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    return float(acc)


def federated_average(x_train, y_train, x_test, y_test, clients=3, rounds=3, full_data_per_client=False):
    len_xtrain = len(x_train)
    if full_data_per_client:
        shards = []
        for client_id in range(clients):
            shards.append(np.arange(len_xtrain))
    else:
        idx = np.random.permutation(len_xtrain)
        shards = np.array_split(idx, clients)

    global_model = make_model()
    global_weights = global_model.get_weights()

    for i in range(rounds):
        client_weights = []
        for shard in shards:
            local = make_model()
            local.set_weights(global_weights)
            local.fit(x_train[shard], y_train[shard], epochs=1, batch_size=64, verbose=0)
            client_weights.append(local.get_weights())

        new_weights = []
        for weights in zip(*client_weights):
            new_weights.append(np.mean(weights, axis=0))
        global_weights = new_weights

    global_model.set_weights(global_weights)
    loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
    return float(acc)


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

    acc_c = train_centralized(train_images, train_labels, test_images, test_labels)
    acc_f = federated_average(train_images, train_labels, test_images, test_labels, clients=clients, full_data_per_client=full_per_client)

    label.config(text=f"Centralized acc: {acc_c:.4f}    Federated acc: {acc_f:.4f}")
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
        run_demo(btn, lbl)

    btn.config(command=on_click)

    root.mainloop()


if __name__ == "__main__":
    main()

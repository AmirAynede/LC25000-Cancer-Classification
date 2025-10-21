# LC25000 Cancer Classification Project
# Author: Amir Aynede
# File: plot.py
# Description: Plot training and validation metrics from JSON logs

import json
import matplotlib.pyplot as plt
import os

def plot_training_metrics(json_path):
    with open(json_path, "r") as f:
        metrics = json.load(f)

    epochs = range(1, len(metrics["train_loss"]) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    loss_path = os.path.join("outputs", "training_loss.png")
    plt.savefig(loss_path)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, metrics["train_acc"], label="Train Accuracy")
    plt.plot(epochs, metrics["val_acc"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    acc_path = os.path.join("outputs", "training_accuracy.png")
    plt.savefig(acc_path)
    plt.show()

    print(f"Saved plots to: {loss_path} and {acc_path}")

if __name__ == "__main__":
    json_path = "results/training_metrics_latest.json"  # replace with your actual path if needed
    plot_training_metrics(json_path)

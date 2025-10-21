# LC25000 Cancer Classification Project
# Author: Amir Aynede
# File: animate_training_curves.py
# Description: Animated training and validation metric visualization

import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_training(json_path="results/training_metrics_latest.json"):
    with open(json_path, "r") as f:
        metrics = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def update(i):
        ax1.clear()
        ax2.clear()
        epochs = range(1, i + 2)
        ax1.plot(epochs, metrics["train_loss"][:i+1], label="Train Loss")
        ax1.plot(epochs, metrics["val_loss"][:i+1], label="Validation Loss")
        ax1.set_title("Loss Over Epochs")
        ax1.legend()
        ax2.plot(epochs, metrics["train_acc"][:i+1], label="Train Accuracy")
        ax2.plot(epochs, metrics["val_acc"][:i+1], label="Validation Accuracy")
        ax2.set_title("Accuracy Over Epochs")
        ax2.legend()

    ani = FuncAnimation(fig, update, frames=len(metrics["train_loss"]), interval=400, repeat=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    animate_training()

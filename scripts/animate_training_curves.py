import glob
import os
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Find the latest training_metrics_*.json file
def get_latest_metrics_file():
    metrics_files = glob.glob("results/training_metrics_*.json")
    if not metrics_files:
        raise FileNotFoundError("No training_metrics_*.json files found in results/")
    return max(metrics_files, key=os.path.getctime)

def animate_training_curves():
    metrics_path = get_latest_metrics_file()
    print(f"Using metrics file: {metrics_path}")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    train_losses = metrics["train_losses"]
    val_losses = metrics["val_losses"]
    train_accuracies = metrics["train_accuracies"]
    val_accuracies = metrics["val_accuracies"]
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def animate(i):
        ax1.clear()
        ax2.clear()
        ax1.plot(epochs[:i+1], train_losses[:i+1], 'b-', label='Train Loss')
        ax1.plot(epochs[:i+1], val_losses[:i+1], 'r-', label='Val Loss')
        ax1.set_title("Loss Over Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.set_xlim(1, len(epochs))
        ax1.set_ylim(0, max(max(train_losses), max(val_losses)) * 1.1)

        ax2.plot(epochs[:i+1], train_accuracies[:i+1], 'b-', label='Train Acc')
        ax2.plot(epochs[:i+1], val_accuracies[:i+1], 'r-', label='Val Acc')
        ax2.set_title("Accuracy Over Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.set_xlim(1, len(epochs))
        ax2.set_ylim(0, 1.05)

    ani = FuncAnimation(fig, animate, frames=len(epochs), interval=200, repeat=False)
    plt.show()

if __name__ == "__main__":
    animate_training_curves() 
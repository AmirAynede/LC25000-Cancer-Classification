import matplotlib.pyplot as plt
import json
import sys
import os

plt.style.use('dark_background')  # Set dark background

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, output_dir="outputs", prefix="training_curve"):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # === Plot Loss ===
    axs[0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axs[0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axs[0].set_title("Training and Validation Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # === Plot Accuracy ===
    axs[1].plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    axs[1].plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    axs[1].set_title("Training and Validation Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{prefix}_combined.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"✅ Saved combined loss/accuracy plot to {plot_path}")

def load_metrics(json_path):
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        sys.exit(1)
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    return metrics

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m scripts.plot path/to/training_metrics.json")
        sys.exit(1)

    metrics_path = sys.argv[1]
    metrics = load_metrics(metrics_path)

    plot_training_curves(
        metrics["train_losses"],
        metrics["val_losses"],
        metrics["train_accuracies"],
        metrics["val_accuracies"],
        output_dir="outputs",
        prefix=os.path.splitext(os.path.basename(metrics_path))[0]
    )
import os
import torch
from glob import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from scripts.dataset import prepare_dataloaders
from scripts.model import build_model


def find_latest_model(model_dir="saved_models", prefix="resnet18_lc25000"):
    model_files = glob(os.path.join(model_dir, f"{prefix}_*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir} with prefix {prefix}")
    latest_model = max(model_files, key=os.path.getctime)
    print(f"📂 Using latest model: {latest_model}")
    return latest_model


def evaluate_model(model_path=None, data_path="data/lc25000_split", batch_size=32):
    # === Device Setup ===
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    # === Load Data (Test Only) ===
    _, _, test_loader, class_names = prepare_dataloaders(data_path, batch_size=batch_size)
    print("📁 Detected classes in test set:", class_names) #checking if prepare_dataloaders() correctly sees all 5 class
                                                            #folders under data/lc25000_split/test/.
    # === Load Model ===
    if model_path is None:
        model_path = find_latest_model()

    model = build_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []

    # === Run Inference ===
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # === Classification Report ===
    print("\n📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # === Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")  # <-- Save image to outputs/
    plt.show()


if __name__ == "__main__":
    evaluate_model()
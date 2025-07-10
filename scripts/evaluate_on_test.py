"""
Clean evaluation script for the LC25000 classification project.

This script is an improved and updated version of the original 'evaluate.py'.
It includes fixes for dataset path handling and model class mismatch issues,
ensuring compatibility with the current dataset structure and model configuration.

IMPORTANT NOTES for users:
- This script is designed to work with the latest data splits located under
  'data/lc25000_split/', contains three main folders:
    - train/
    - val/
    - test/

- Each of these folders contains subfolders named by class labels, for example:
    train/
      ├── colon_aca/
      ├── colon_n/
      ├── lung_aca/
      ├── lung_n/
      └── lung_scc/

    val/
      ├── colon_aca/
      ├── colon_n/
      ├── lung_aca/
      ├── lung_n/
      └── lung_scc/

    test/
      ├── colon_aca/
      ├── colon_n/
      ├── lung_aca/
      ├── lung_n/
      └── lung_scc/

- Each class folder contains image files belonging to that class, properly organized for
  the torchvision ImageFolder dataset loader to correctly assign labels.

- This folder organization enables the dataset loader to:
    1. Automatically detect the classes from the folder names.
    2. Load images and assign correct labels during training, validation, and testing.
  Each of these folders should have subdirectories named after the class labels,
  containing the respective images.
  
- The script expects the number of classes to match the subfolder count in these
  directories. Maintaining this folder structure allows consistent loading,
  evaluation, and reproducibility of results.
- The original 'evaluate.py' is preserved separately to maintain reproducibility of
  previous results and experiments.
- Use this script for new evaluations and experiments where the dataset or model
  architecture may have changed.
- Retaining both versions allows comparing evaluation results from different
  project stages and ensures transparency.
- If your goal is strict reproducibility of past results, run the original 'evaluate.py'.
- Once confident in the new evaluation pipeline, you may replace the old script,
  but keep backups or use version control to track changes.

This approach helps maintain project clarity, versioning, and reproducibility for
all users and future work.
"""

import os
import torch
from glob import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from scripts.dataset import prepare_dataloaders
from scripts.model import build_model


def find_latest_model(model_dir="saved_models", prefix="resnet18_lc25000"):
    model_files = glob(os.path.join(model_dir, f"{prefix}_*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir} with prefix {prefix}")
    latest_model = max(model_files, key=os.path.getctime)
    print(f"📂 Using latest model: {latest_model}")
    return latest_model


def evaluate_model(model_path=None, data_path="data/lc25000_split", batch_size=32, num_classes=None):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    # Prepare dataloaders; get test loader and classes only
    test_data_path = os.path.join(data_path, "test")
    _, _, test_loader, class_names = prepare_dataloaders(test_data_path, batch_size=batch_size)

    # If num_classes not specified, infer from folders
    if num_classes is None:
        num_classes = len(class_names)

    # Load model path if not given
    if model_path is None:
        model_path = find_latest_model()

    # Build model with specified num_classes (must match checkpoint)
    model = build_model(num_classes=num_classes).to(device)

    # Load checkpoint
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    filenames = []

    # Run inference on test set
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            # Handle Subset or ImageFolder
            dataset = test_loader.dataset
            if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'samples'):
                # If using Subset
                samples = dataset.dataset.samples
                indices = dataset.indices
                batch_indices = indices[len(filenames):len(filenames) + len(labels)]
                filenames.extend([os.path.abspath(samples[i][0]) for i in batch_indices])
            elif hasattr(dataset, 'samples'):
                # If using ImageFolder directly
                batch_indices = range(len(filenames), len(filenames) + len(labels))
                filenames.extend([os.path.abspath(dataset.samples[i][0]) for i in batch_indices])
            else:
                filenames.extend([''] * len(labels))

    # Print classification report
    print("\n📊 Classification Report:\n")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # Save classification report to file
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/classification_report.txt", "w") as f:
        f.write(report)
    print("✅ Saved classification report to outputs/classification_report.txt")

    # Save raw predictions to CSV
    pred_df = pd.DataFrame({
        'filename': filenames,
        'true_label_idx': y_true,
        'true_label': [class_names[i] for i in y_true],
        'predicted_label_idx': y_pred,
        'predicted_label': [class_names[i] for i in y_pred]
    })
    pred_df.to_csv('outputs/test_predictions.csv', index=False)
    print('✅ Saved raw predictions to outputs/test_predictions.csv')

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    evaluate_model(num_classes=5)
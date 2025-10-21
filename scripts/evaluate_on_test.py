# LC25000 Cancer Classification Project
# Author: Amir Aynede
# File: evaluate_on_test.py
# Description: Evaluate model performance on test dataset and save reports

import os
import csv
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import build_model

def evaluate(model, test_loader, classes, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=classes, digits=2)
    print(report)
    return all_labels, all_preds, report

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = "data/lc25000_split/train"
    val_dir = "data/lc25000_split/val"
    test_dir = "data/lc25000_split/test"

    _, _, test_loader, classes = get_dataloaders(train_dir, val_dir, test_dir, batch_size=32)
    model = build_model(num_classes=len(classes))
    model.load_state_dict(torch.load("saved_models/best_model_latest.pth", map_location=device))
    model = model.to(device)

    all_labels, all_preds, report = evaluate(model, test_loader, classes, device)

    os.makedirs("results", exist_ok=True)
    with open("results/classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()

    print("Saved classification report and confusion matrix.")

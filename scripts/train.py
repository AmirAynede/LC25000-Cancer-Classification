# LC25000 Cancer Classification Project
# Author: Amir Aynede
# File: train.py
# Description: Train the ResNet18 model on the LC25000 dataset

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from dataset import get_dataloaders
from model import build_model

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs=30):
    metrics = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total = 0

            loop = tqdm(loader, desc=f"{phase} phase", leave=False)
            for inputs, labels in loop:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total
            metrics[f"{phase}_loss"].append(epoch_loss)
            metrics[f"{phase}_acc"].append(epoch_acc.item())

            print(f"{phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        scheduler.step(metrics["val_loss"][-1])

    model.load_state_dict(best_model_wts)
    print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")
    return model, metrics

if __name__ == "__main__":
    train_dir = "data/lc25000_split/train"
    val_dir = "data/lc25000_split/val"
    test_dir = "data/lc25000_split/test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, classes = get_dataloaders(train_dir, val_dir, test_dir, batch_size=32)
    model = build_model(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.1, verbose=True)

    model, metrics = train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, device, num_epochs=30)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"results/training_metrics_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f)

    os.makedirs("saved_models", exist_ok=True)
    model_path = f"saved_models/best_model_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved metrics to: {json_path}")

import torch.nn as nn
from torchvision import models

def build_model(num_classes):
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)

    # Freeze early layers (optional)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer for our custom classification task
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
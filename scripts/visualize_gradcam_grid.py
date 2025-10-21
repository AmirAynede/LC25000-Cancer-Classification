# LC25000 Cancer Classification Project
# Author: Amir Aynede
# File: visualize_gradcam_grid.py
# Description: Display grid comparison of original and Grad-CAM images

import os
import random
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import build_model
from gradcam import GradCAM

def visualize_gradcam_grid(csv_path="results/test_predictions.csv", model_path="saved_models/best_model.pth",
                           n_images=4, cols=2, only_misclassified=True, output_path="outputs/gradcam_grid.png"):

    df = pd.read_csv(csv_path)
    if only_misclassified:
        df = df[df["true_label"] != df["pred_label"]]
    sample_df = df.sample(n=min(n_images, len(df)), random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    target_layer = list(model.layer4.children())[-1]
    gradcam = GradCAM(model, target_layer)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    fig, axes = plt.subplots(len(sample_df), 2, figsize=(8, 4 * len(sample_df)))
    if len(sample_df) == 1:
        axes = np.expand_dims(axes, 0)

    for i, (_, row) in enumerate(sample_df.iterrows()):
        image = Image.open(row["image_path"]).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
        class_idx = torch.argmax(output, dim=1).item()
        cam = gradcam.generate(input_tensor, class_idx)
        cam_resized = cv2.resize(cam, (224, 224))
        cam_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        overlay = np.float32(cam_color) / 255 + np.float32(image) / 255
        overlay = overlay / np.max(overlay)

        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Original: {row['true_label']}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f"Grad-CAM: Pred {row['pred_label']}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output_path)
    plt.show()
    print(f"Saved Grad-CAM grid to {output_path}")

if __name__ == "__main__":
    visualize_gradcam_grid()

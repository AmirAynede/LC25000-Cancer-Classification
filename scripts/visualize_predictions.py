# LC25000 Cancer Classification Project
# Author: Amir Aynede
# File: visualize_predictions.py
# Description: Visualize random test predictions as an image grid

import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def visualize_predictions(csv_path="results/test_predictions.csv", n_images=9, cols=3, output_path="outputs/prediction_grid.png"):
    df = pd.read_csv(csv_path)
    sample_df = df.sample(n=n_images, random_state=42)

    rows = n_images // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()

    for i, (idx, row) in enumerate(sample_df.iterrows()):
        img = mpimg.imread(row["image_path"])
        axes[i].imshow(img)
        color = "green" if row["true_label"] == row["pred_label"] else "red"
        axes[i].set_title(f"T: {row['true_label']} | P: {row['pred_label']}", color=color)
        axes[i].axis("off")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output_path)
    plt.show()
    print(f"Saved prediction grid to {output_path}")

if __name__ == "__main__":
    visualize_predictions()

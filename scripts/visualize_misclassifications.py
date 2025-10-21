# LC25000 Cancer Classification Project
# Author: Amir Aynede
# File: visualize_misclassifications.py
# Description: Display grid of misclassified test samples

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def visualize_misclassifications(csv_path="results/test_predictions.csv", n_images=9, cols=3, output_path="outputs/misclassified_grid.png"):
    df = pd.read_csv(csv_path)
    misclassified = df[df["true_label"] != df["pred_label"]]
    if misclassified.empty:
        print("No misclassifications found!")
        return

    sample_df = misclassified.sample(n=min(n_images, len(misclassified)), random_state=42)
    rows = n_images // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()

    for i, (idx, row) in enumerate(sample_df.iterrows()):
        img = mpimg.imread(row["image_path"])
        axes[i].imshow(img)
        axes[i].set_title(f"T: {row['true_label']} | P: {row['pred_label']}", color="red")
        axes[i].axis("off")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(output_path)
    plt.show()
    print(f"Saved misclassification grid to {output_path}")

if __name__ == "__main__":
    visualize_misclassifications()

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

def visualize_predictions(csv_path, n_images=9, cols=3, output_path=None):
    df = pd.read_csv(csv_path)
    N = min(n_images, len(df))
    sample = df.sample(N, random_state=42)
    rows = (N + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))

    for ax, (_, row) in zip(axes.flatten(), sample.iterrows()):
        filename = row['filename']
        if isinstance(filename, str) and os.path.exists(filename):
            img = Image.open(filename)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
        color = 'green' if row['true_label'] == row['predicted_label'] else 'red'
        ax.set_title(f"True: {row['true_label']}\nPred: {row['predicted_label']}", color=color, fontsize=12)
        ax.axis('off')
    for ax in axes.flatten()[len(sample):]:
        ax.axis('off')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"✅ Saved grid plot to {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize predictions as an image grid.")
    parser.add_argument('--csv_path', type=str, default='outputs/test_predictions.csv', help='Path to predictions CSV file')
    parser.add_argument('--n_images', type=int, default=9, help='Number of images to display')
    parser.add_argument('--cols', type=int, default=3, help='Number of columns in the grid')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the grid image (optional)')
    args = parser.parse_args()
    visualize_predictions(args.csv_path, args.n_images, args.cols, args.output_path) 
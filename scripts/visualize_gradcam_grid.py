import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
from scripts.gradcam import main as gradcam_main

def visualize_gradcam_grid(csv_path, model_path, n_images=4, cols=2, only_misclassified=True, output_path=None):
    df = pd.read_csv(csv_path)
    if only_misclassified:
        df = df[df['true_label'] != df['predicted_label']]
    N = min(n_images, len(df))
    if N == 0:
        print("No images to show!")
        return
    sample = df.sample(N, random_state=42)
    rows = (N + cols - 1) // cols

    fig, axes = plt.subplots(rows, 2*cols, figsize=(4*2*cols, 4*rows))
    axes = axes.flatten()

    for i, (_, row) in enumerate(sample.iterrows()):
        filename = row['filename']
        if isinstance(filename, str) and os.path.exists(filename):
            # Original image
            img = Image.open(filename)
            axes[2*i].imshow(img)
            axes[2*i].set_title(f"True: {row['true_label']}\nPred: {row['predicted_label']}", color='red', fontsize=12)
            axes[2*i].axis('off')

            # Grad-CAM (call gradcam.py as a function)
            gradcam_img_path = os.path.join("outputs", os.path.basename(filename).split('.')[0] + f"_tmpgrid_gradcam_{i}.jpg")
            gradcam_main(filename, model_path, "layer4.1.conv2", 5, "outputs")
            # The gradcam.py script saves as outputs/<image>_gradcam.jpg
            gradcam_saved_path = os.path.join("outputs", os.path.basename(filename).split('.')[0] + "_gradcam.jpg")
            if os.path.exists(gradcam_saved_path):
                gradcam_img = Image.open(gradcam_saved_path)
                axes[2*i+1].imshow(gradcam_img)
                axes[2*i+1].set_title("Grad-CAM Heatmap")
                axes[2*i+1].axis('off')
                # Optionally, copy to a temp name for clarity
                gradcam_img.save(gradcam_img_path)
            else:
                axes[2*i+1].text(0.5, 0.5, 'Grad-CAM not found', ha='center', va='center')
                axes[2*i+1].axis('off')
        else:
            axes[2*i].text(0.5, 0.5, 'Image not found', ha='center', va='center')
            axes[2*i].axis('off')
            axes[2*i+1].axis('off')

    for ax in axes[2*N:]:
        ax.axis('off')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"✅ Saved Grad-CAM grid to {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Grad-CAM heatmaps in a grid.")
    parser.add_argument('--csv_path', type=str, default='outputs/test_predictions.csv', help='Path to predictions CSV file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--n_images', type=int, default=4, help='Number of images to display')
    parser.add_argument('--cols', type=int, default=2, help='Number of columns in the grid')
    parser.add_argument('--only_misclassified', action='store_true', help='Show only misclassified images')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the grid image (optional)')
    args = parser.parse_args()
    visualize_gradcam_grid(args.csv_path, args.model_path, args.n_images, args.cols, args.only_misclassified, args.output_path) 